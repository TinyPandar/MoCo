import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score
from torch import optim
from torchvision import transforms
from tqdm import tqdm
from MoCo import MoCo
from torchvision.datasets import CIFAR10
from torch.utils.data import DataLoader


def train():
    train_dataset = CIFAR10('datasets', train=True, download=True, transform=transforms.ToTensor())
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    model = MoCo(momentum=0.01, dim=128, T=0.07, Q=64)
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
    )
    augmentation = [
        transforms.RandomResizedCrop(224, scale=(0.2, 1.0), antialias=True),
        transforms.RandomGrayscale(p=0.2),
        transforms.ColorJitter(0.4, 0.4, 0.4, 0.4),
        transforms.RandomHorizontalFlip(),
        normalize,
    ]
    random_transform = transforms.Compose(augmentation)

    optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)
    criterion = torch.nn.CrossEntropyLoss()

    # training
    model.train()
    for epoch in tqdm(range(100)):
        # dataloader只能处理tensor。dataset加了transform
        for img, _ in tqdm(train_loader):
            # 一开始在想怎么用一个dataloader读两个数据，后来仔细看了论文发现是两次变换
            img_q = random_transform(img)
            img_k = random_transform(img)

            out, label = model(img_q, img_k)

            loss = criterion(out, label)
            acc = accuracy_score(out.argmax(), label)
            print(f"loss:{loss.item()} acc:{acc}")
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()


if __name__ == '__main__':
    train()
