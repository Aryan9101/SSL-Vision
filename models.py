import torch
import torch.nn as nn

from timm.models.layers import trunc_normal_
from timm.models.vision_transformer import PatchEmbed, Block

class ViTLayer(nn.Module):
    def __init__(self, num_heads, embed_dim, encoder_mlp_hidden, dropout=0.1):
        super().__init__()
        self.layernorm1 = nn.LayerNorm(embed_dim)
        self.msa = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)
        self.attn_dropout = nn.Dropout(dropout)
        self.layernorm2 = nn.LayerNorm(embed_dim)
        self.mlp = nn.Sequential(nn.Linear(embed_dim, encoder_mlp_hidden), 
                                 nn.GELU(),
                                 nn.Dropout(dropout),
                                 nn.Linear(encoder_mlp_hidden, embed_dim),
                                 nn.Dropout(dropout))

    def forward(self, x):
        # keep track of shapes
        norm = self.layernorm1(x)
        attn, _ = self.msa(norm, norm, norm)
        attn = self.attn_dropout(attn)
        x = x + attn
        norm = self.layernorm2(x)
        x = x + self.mlp(norm)
        return x

class ViT(nn.Module):
    def __init__(self, patch_dim, image_dim, num_layers, num_heads, embed_dim, encoder_mlp_hidden, num_classes, dropout):
        super().__init__()
        self.num_layers = num_layers
        self.patch_dim = patch_dim
        self.image_dim = image_dim
        self.input_dim = self.patch_dim * self.patch_dim * 3

        self.patch_embedding = nn.Linear(self.input_dim, embed_dim)
        self.position_embedding = nn.Parameter(torch.zeros(1, (image_dim // patch_dim) ** 2 + 1, embed_dim))
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.embedding_dropout = nn.Dropout(dropout)

        self.encoder_layers = nn.ModuleList([])
        for i in range(num_layers):
            self.encoder_layers.append(ViTLayer(num_heads, embed_dim, encoder_mlp_hidden, dropout))

        self.mlp_head = nn.Linear(embed_dim, num_classes)
        self.layernorm = nn.LayerNorm(embed_dim)

    def forward(self, images):
        h = w = self.image_dim // self.patch_dim
        N = images.shape[0]        
        images = images.reshape(N, 3, h, self.patch_dim, w, self.patch_dim)
        images = torch.einsum("nchpwq -> nhwpqc", images)
        patches = images.reshape(N, h * w, self.input_dim)

        patch_embeddings = self.patch_embedding(patches)
        patch_embeddings = torch.cat([torch.tile(self.cls_token, (N, 1, 1)), 
                                      patch_embeddings], dim=1)
        out = patch_embeddings + torch.tile(self.position_embedding, (N, 1, 1))
        out = self.embedding_dropout(out)

        for i in range(self.num_layers):
            out = self.encoder_layers[i](out)

        cls_head = self.layernorm(torch.squeeze(out[:, 0, :], dim=1))
        logits = self.mlp_head(cls_head)
        return logits

class MAE_Timm(nn.Module):
    def __init__(self, patch_dim, image_dim,
                 encoder_num_layers, encoder_num_heads, encoder_embed_dim, 
                 decoder_num_layers, decoder_num_heads, decoder_embed_dim,
                 dropout, device):
        super().__init__()
        self.device = device
        self.patch_dim = patch_dim
        self.image_dim = image_dim
        self.num_patches = (image_dim // patch_dim) ** 2
        self.input_dim = self.patch_dim * self.patch_dim * 3

        self.patch_embedding = PatchEmbed(self.image_dim, self.patch_dim, 3, encoder_embed_dim)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, encoder_embed_dim))
        self.encoder_position_embedding = nn.Parameter(torch.zeros(1, self.num_patches + 1, encoder_embed_dim))

        self.encoder_num_layers = encoder_num_layers
        self.encoder_layers = nn.ModuleList([])
        for _ in range(self.encoder_num_layers):
            self.encoder_layers.append(Block(encoder_embed_dim, encoder_num_heads, 4, qkv_bias=False, drop=dropout, attn_drop=dropout))

        self.encoder_layernorm = nn.LayerNorm(encoder_embed_dim)

        self.decoder_embedding = nn.Linear(encoder_embed_dim, decoder_embed_dim, bias=True)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_embed_dim))
        self.decoder_position_embedding = nn.Parameter(torch.zeros(1, self.num_patches + 1, decoder_embed_dim))

        self.decoder_num_layers = decoder_num_layers
        self.decoder_layers = nn.ModuleList([])
        for _ in range(self.decoder_num_layers):
            self.decoder_layers.append(Block(decoder_embed_dim, decoder_num_heads, 4, qkv_bias=False, drop=dropout, attn_drop=dropout))

        self.decoder_layernorm = nn.LayerNorm(decoder_embed_dim)
        self.image_projection = nn.Linear(decoder_embed_dim, self.input_dim)

        self.init_weights()

    def patchify(self, images):
        N = images.shape[0]
        h = w = self.image_dim // self.patch_dim
        images = images.reshape(N, 3, h, self.patch_dim, w, self.patch_dim)
        images = torch.einsum("nchpwq -> nhwpqc", images)
        patches = images.reshape(N, self.num_patches, self.input_dim)
        return patches

    def init_weights(self):
        w = self.patch_embedding.proj.weight.data
        torch.nn.init.xavier_uniform_(w.view([w.shape[0], -1]))

        torch.nn.init.normal_(self.encoder_position_embedding, std=.02)
        torch.nn.init.normal_(self.decoder_position_embedding, std=.02)
        torch.nn.init.normal_(self.cls_token, std=.02)
        torch.nn.init.normal_(self.mask_token, std=.02)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            # we use xavier_uniform following official JAX ViT:
            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def encode(self, images, mask_ratio):
        patch_embeddings = self.patch_embedding(images) + self.encoder_position_embedding[:, 1:]
        N, _, D = patch_embeddings.shape
        
        rand = torch.rand(N, self.num_patches).to(self.device)
        idx_shuffle = torch.argsort(rand, dim=1)
        idx_unshuffle = torch.argsort(idx_shuffle, dim=1)
        
        keep = int(self.num_patches * (1 - mask_ratio))
        
        patch_embeddings = torch.gather(patch_embeddings, dim=1, index=idx_shuffle.unsqueeze(-1).tile(1, 1, D))[:, :keep]

        mask = torch.ones(N, self.num_patches).to(self.device)
        mask[:, keep:] = 0
        mask = torch.gather(mask, dim=1, index=idx_unshuffle)

        class_tokens = torch.tile(self.cls_token, (N, 1, 1)) + self.encoder_position_embedding[:, :1]
        out = torch.cat([class_tokens, patch_embeddings], dim=1)

        for i in range(self.encoder_num_layers):
            out = self.encoder_layers[i](out)

        out = self.encoder_layernorm(out)

        return patch_embeddings, mask, idx_unshuffle

    def decode(self, patches, idx_unshuffle):
        patch_embeddings = self.decoder_embedding(patches)

        embedding_cls = patch_embeddings[:, :1]
        embedding_image = patch_embeddings[:, 1:]
        N, L, D = embedding_image.shape
        embedding_image = torch.cat([embedding_image, torch.tile(self.mask_token, (N, self.num_patches - L, 1))], dim=1)
        embedding_image = torch.gather(embedding_image, dim=1, index=idx_unshuffle.unsqueeze(-1).tile(1, 1, D))
        patch_embeddings = torch.cat([embedding_cls, embedding_image], dim=1)
        out = patch_embeddings + self.decoder_position_embedding

        for i in range(self.decoder_num_layers):
            out = self.decoder_layers[i](out)

        out = self.decoder_layernorm(out)[:, 1:]
        image_patches = self.image_projection(out)

        return image_patches

    def loss(self, images, pred_patches, mask):
        patches = self.patchify(images)
        mean = patches.mean(dim=-1, keepdim=True)
        var = patches.var(dim=-1, keepdim=True)
        patches = (patches - mean) / (var + 1.e-6)**.5

        loss = (patches - pred_patches) ** 2
        loss = loss.mean(dim=-1)  # [N, L], mean loss per patch

        mask = 1 - mask
        loss = (loss * mask).sum() / mask.sum()  # mean loss on removed patches
        return loss

    def recover_reconstructed(self, images, mask_ratio):
        image_patches = self.patchify(images)
        
        patches, mask, idx_unshuffle = self.encode(images, mask_ratio)

        pred_patches = self.decode(patches, idx_unshuffle)
        
        N = images.shape[0]
        h = w = self.image_dim // self.patch_dim
        
        mask = mask.unsqueeze(-1)

        masked_images = mask * image_patches
        masked_images = masked_images.reshape(N, h, w, self.patch_dim, self.patch_dim, 3)
        masked_images = torch.einsum("nhwpqc -> nchpwq", masked_images).reshape(N, 3, self.image_dim, self.image_dim)

        mean = image_patches.mean(dim=-1, keepdim=True)
        var = image_patches.var(dim=-1, keepdim=True)
        reconstructed = pred_patches * (var + 1.e-6)**.5 + mean
        # reconstructed = mask * image_patches + (1 - mask) * pred_patches
        reconstructed = reconstructed.reshape(N, h, w, self.patch_dim, self.patch_dim, 3)
        reconstructed = torch.einsum("nhwpqc -> nchpwq", reconstructed).reshape(N, 3, self.image_dim, self.image_dim)

        return masked_images, reconstructed

    def forward(self, images, mask_ratio=0.0):
        patches, mask, idx_unshuffle = self.encode(images, mask_ratio)
        pred_patches = self.decode(patches, idx_unshuffle)
        loss = self.loss(images, pred_patches, mask)

        return loss