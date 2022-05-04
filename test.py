import torch

import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset

import matplotlib.pyplot as plt
import numpy as np

from models import MAE_Timm

if torch.cuda.is_available():
    device = torch.device("cuda:0")
    print("Running on a GPU")
else:
    device = torch.device("cpu")
    print("Running on a CPU")

checkpoint = torch.load(f"./SSL-Vision/mae_timm.pth")
patch_dim = checkpoint['patch_dim']
image_dim = checkpoint['image_dim']
encoder_num_layers = checkpoint['encoder_num_layers']
encoder_num_heads = checkpoint['encoder_num_heads']
encoder_embed_dim = checkpoint['encoder_embed_dim']
encoder_mlp_hidden = encoder_embed_dim * 4
decoder_num_layers = checkpoint['decoder_num_layers']
decoder_num_heads = checkpoint['decoder_num_heads']
decoder_embed_dim = checkpoint['decoder_embed_dim']
decoder_mlp_hidden = decoder_embed_dim * 4
dropout = checkpoint['dropout']

mae = MAE_Timm(patch_dim, image_dim, 
               encoder_num_layers, encoder_num_heads, encoder_embed_dim,
               decoder_num_heads, decoder_num_layers, decoder_embed_dim,
               dropout, device).to(device)
mae.load_state_dict(checkpoint['vit_state_dict'])

cifar10_mean = torch.tensor([0.49139968, 0.48215827, 0.44653124])
cifar10_std = torch.tensor([0.24703233, 0.24348505, 0.26158768])

class Cifar10Dataset(Dataset):
    def __init__(self, train):
        self.transform = transforms.Compose([
                                                transforms.ToTensor(),
                                                transforms.Normalize(cifar10_mean, cifar10_std)
                                            ])
        self.dataset = torchvision.datasets.CIFAR10(root='./SSL-Vision/data', 
                                                    train=train,
                                                    download=True)
    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        img, label = self.dataset[idx]
        img = self.transform(img)
        return img, label

testset = Cifar10Dataset(False)
testloader = DataLoader(testset, batch_size=16, shuffle=False, num_workers=2)

image_samples, _ = next(iter(testloader))
masked_images, reconstructed = mae.recover_reconstructed(image_samples.to(device), mask_ratio=0.75)
_ , recon_no_mask = mae.recover_reconstructed(image_samples.to(device), mask_ratio=0.00)
image_samples = image_samples.permute(0, 2, 3, 1).cpu()
masked_images = masked_images.permute(0, 2, 3, 1).detach().cpu()
reconstructed = reconstructed.permute(0, 2, 3, 1).detach().cpu()
recon_no_mask = recon_no_mask.permute(0, 2, 3, 1).detach().cpu()
image_samples = torch.clip((image_samples * cifar10_std + cifar10_mean) * 255, 0, 255).int()
masked_images = torch.clip((masked_images * cifar10_std + cifar10_mean) * 255, 0, 255).int()
reconstructed = torch.clip((reconstructed * cifar10_std + cifar10_mean) * 255, 0, 255).int()
recon_no_mask = torch.clip((recon_no_mask * cifar10_std + cifar10_mean) * 255, 0, 255).int()

fig, axes = plt.subplots(8, 8, figsize=(10, 10))
axes = np.array(axes).flatten()
for i, ax in enumerate(axes[0::4]):
    ax.imshow(image_samples[i].numpy())
    ax.axis('off')
for i, ax in enumerate(axes[1::4]):
    ax.imshow(masked_images[i].numpy())
    ax.axis('off')
for i, ax in enumerate(axes[2::4]):
    ax.imshow(reconstructed[i].numpy())
    ax.axis('off')
for i, ax in enumerate(axes[3::4]):
    ax.imshow(recon_no_mask[i].numpy())
    ax.axis('off')
fig.tight_layout()
fig.savefig(f"./SSL-Vision/mae_results/test.png")
plt.close(fig)