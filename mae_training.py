import torch
import torch.nn as nn
import torch.optim as optim

import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset

import transformers

import matplotlib.pyplot as plt
import numpy as np
import tqdm

from models import *

import wandb

wandb.init(project="ssl-vision", entity="aryan9101")

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
                                                transforms.RandomResizedCrop(size=32, scale=(0.2, 1.0), interpolation=3),
                                                transforms.RandomHorizontalFlip(),
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

batch_size = 128 * torch.cuda.device_count()

trainset = Cifar10Dataset(True)
trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=2)

testset = Cifar10Dataset(False)
testloader = DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=2)

mae = get_mae_base().to(device)
mae = torch.nn.DataParallel(mae)

mask_ratio = 0.75
learning_rate = 1.5e-4 * batch_size / 256
num_epochs = 400
warmup_fraction = 0.1
weight_decay = 0.05

# total_steps = math.ceil(len(trainset) / batch_size) * num_epochs
total_steps = num_epochs
warmup_steps = total_steps * warmup_fraction
criterion = nn.CrossEntropyLoss()
optimizer = optim.AdamW(mae.parameters(), lr=learning_rate, betas=(0.9, 0.95), weight_decay=weight_decay)
scheduler = transformers.get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, 
                                                         num_training_steps=total_steps)

train_losses = []
test_losses = []
for epoch in range(num_epochs):
    train_loss = 0.0
    train_acc = 0.0
    train_total = 0
    mae.train()
    for images, labels in tqdm.tqdm(trainloader):
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        loss = mae(images, mask_ratio).sum()
        loss.backward()
        optimizer.step()

        train_loss += loss.item() * images.shape[0]
        train_total += images.shape[0]
    train_loss = train_loss / train_total
    train_losses.append(train_loss)

    scheduler.step()
    
    test_loss = 0.0
    test_acc = 0.0
    test_total = 0
    mae.eval()
    with torch.no_grad():
        for images, labels in testloader:
            images = images.to(device)
            labels = labels.to(device)

            loss = mae(images, mask_ratio).sum()

            test_loss += loss.item() * images.shape[0]
            test_total += images.shape[0]
    test_loss = test_loss / test_total
    test_losses.append(test_loss)
    
    print(f'[{epoch + 1:2d}] train loss: {train_loss:.3f} | test_loss: {test_loss:.3f}')

    if test_loss <= min(test_losses):
        torch.save({'mae_state_dict' : mae.module.state_dict()}, f"./SSL-Vision/mae_timm.pth")
    
    if (epoch + 1) % 25 == 0:
        with torch.no_grad():
            image_samples, _ = next(iter(testloader))
            masked_images, reconstructed = mae.module.recover_reconstructed(image_samples.to(device), mask_ratio)
            image_samples = image_samples.permute(0, 2, 3, 1).cpu()
            masked_images = masked_images.permute(0, 2, 3, 1).detach().cpu()
            reconstructed = reconstructed.permute(0, 2, 3, 1).detach().cpu()
            image_samples = torch.clip((image_samples * cifar10_std + cifar10_mean) * 255, 0, 255).int()
            masked_images = torch.clip((masked_images * cifar10_std + cifar10_mean) * 255, 0, 255).int()
            reconstructed = torch.clip((reconstructed * cifar10_std + cifar10_mean) * 255, 0, 255).int()

            fig, axes = plt.subplots(6, 6, figsize=(10, 10))
            axes = np.array(axes).flatten()
            for i, ax in enumerate(axes[0::3]):
                ax.imshow(image_samples[i].numpy())
                ax.axis('off')
            for i, ax in enumerate(axes[1::3]):
                ax.imshow(masked_images[i].numpy())
                ax.axis('off')
            for i, ax in enumerate(axes[2::3]):
                ax.imshow(reconstructed[i].numpy())
            fig.tight_layout()
            fig.savefig(f"./SSL-Vision/mae_results-dp/epoch_{epoch + 1}.png")
            plt.close(fig)
    
    wandb.log({"train_loss": train_loss, "test_loss": test_loss, "epoch": epoch + 1})

print('Finished Training')