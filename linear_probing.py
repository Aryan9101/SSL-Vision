import torch
import torch.nn as nn
import torch.optim as optim

import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset

from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import seaborn as sns
import numpy as np

import tqdm
import gc

from models import MAE_Classifier, MAE_Timm

if torch.cuda.is_available():
    device = torch.device("cuda:0")
    print("Running on a GPU")
else:
    device = torch.device("cpu")
    print("Running on a CPU")

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

batch_size = 128

trainset = Cifar10Dataset(True)
trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=2)

testset = Cifar10Dataset(False)
testloader = DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=2)

classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

# MAE config
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

mae_classifier = MAE_Classifier(encoder_embed_dim, len(classes)).to(device)

learning_rate = 1e-3
num_epochs = 10

criterion = nn.CrossEntropyLoss()
optimizer = optim.AdamW(mae_classifier.parameters(), lr=learning_rate)

mae.eval()
train_losses = []
test_losses = []
for epoch in range(num_epochs):
    train_loss = 0.0
    train_acc = 0.0
    train_total = 0
    mae_classifier.train()
    for inputs, labels in tqdm.tqdm(trainloader):
        inputs = inputs.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        with torch.no_grad():
            embeds = mae.embeddings(inputs)
        _, outputs = mae_classifier(embeds)
        loss = criterion(outputs, labels.long())
        loss.backward()
        optimizer.step()

        train_loss += loss.item() * inputs.shape[0]
        train_acc += torch.sum((torch.argmax(outputs, dim=1) == labels)).item()
        train_total += inputs.shape[0]
    train_loss = train_loss / train_total
    train_acc = train_acc / train_total
    train_losses.append(train_loss)
    
    test_loss = 0.0
    test_acc = 0.0
    test_total = 0
    mae_classifier.eval()
    with torch.no_grad():
        for inputs, labels in testloader:
            inputs = inputs.to(device)
            labels = labels.to(device)

            embeds = mae.embeddings(inputs)
            _, outputs = mae_classifier(embeds)
            loss = criterion(outputs, labels.long())

            test_loss += loss.item() * inputs.shape[0]
            test_acc += torch.sum((torch.argmax(outputs, dim=1) == labels)).item()
            test_total += inputs.shape[0]
    test_loss = test_loss / test_total
    test_acc = test_acc / test_total
    test_losses.append(test_loss)
    
    print(f'[{epoch + 1:2d}] train loss: {train_loss:.3f} | train accuracy: {train_acc:.3f} | test_loss: {test_loss:.3f} | test_accuracy: {test_acc:.3f}')

    if min(test_losses[-4:]) > min(test_losses):
        break

print('Finished Training')

gc.collect()
with torch.no_grad():
    mae_latent_train = torch.zeros(len(trainset), encoder_embed_dim)
    mae_labels_train = torch.zeros(len(trainset))
    for i, (inputs, labels) in enumerate(trainloader):
        embeds = mae.embeddings(inputs.to(device))
        latent, _ = mae_classifier(embeds)
        mae_latent_train[i*batch_size:(i+1)*batch_size] = latent
        mae_labels_train[i*batch_size:(i+1)*batch_size] = labels
    mae_latent_test = torch.zeros(len(testset), encoder_embed_dim)
    mae_labels_test = torch.zeros(len(testset))
    for i, (inputs, labels) in enumerate(testloader):
        embeds = mae.embeddings(inputs.to(device))
        latent, _ = mae_classifier(embeds)
        mae_latent_test[i*batch_size:(i+1)*batch_size] = latent
        mae_labels_test[i*batch_size:(i+1)*batch_size] = labels

print("Collected normalized MAE representations.")

mae_latent = torch.cat([mae_latent_train, mae_latent_test])
mae_labels = torch.cat([mae_labels_train, mae_labels_test])

mae_pca = PCA(n_components=128).fit_transform(mae_latent)
mae_embedded = TSNE(n_components=2, perplexity=50, learning_rate='auto', init='random').fit_transform(mae_pca)
mae_class_labels = np.take(np.array(classes), mae_labels.numpy().astype(int))
tsne_plot = sns.scatterplot(x=mae_embedded[:, 0], y=mae_embedded[:, 1], hue=mae_class_labels)
fig = tsne_plot.get_figure()
fig.savefig("./SSL-Vision/mae_results/tsne.png") 

print("T-SNE visualization available.")