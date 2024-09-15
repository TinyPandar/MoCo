import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score
from torch import optim
from torchvision import transforms
from tqdm import tqdm
from MoCo import MoCo
import torchvision.datasets as datasets
from torch.utils.data import DataLoader, DistributedSampler
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import os


class TwoCropsTransform:
    """Take two random crops of one image as the query and key."""

    def __init__(self, base_transform):
        self.base_transform = base_transform

    def __call__(self, x):
        q = self.base_transform(x)
        k = self.base_transform(x)
        return [q, k]


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


def setup_distributed(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    dist.init_process_group("nccl", rank=rank, world_size=world_size)


def train(rank, world_size):
    setup_distributed(rank, world_size)
    torch.cuda.set_device(rank)

    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
    )
    augmentation = [
        transforms.RandomResizedCrop(224, scale=(0.2, 1.0), antialias=True),
        transforms.RandomGrayscale(p=0.2),
        transforms.ColorJitter(0.4, 0.4, 0.4, 0.4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize,
    ]

    train_dataset = datasets.ImageFolder(
        'imagenet/train', TwoCropsTransform(transforms.Compose(augmentation))
    )

    train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank, shuffle=True)
    train_loader = DataLoader(train_dataset, batch_size=64, sampler=train_sampler, drop_last=True)

    model = MoCo(momentum=0.999, dim=128, T=0.07, Q=65536).cuda()
    model = DDP(model, device_ids=[rank])

    optimizer = optim.SGD(model.parameters(), lr=0.0075, momentum=0.9, weight_decay=1e-4)
    criterion = torch.nn.CrossEntropyLoss().cuda()

    model.train()
    for epoch in range(100):
        train_sampler.set_epoch(epoch)
        for batch, (img, _) in enumerate(train_loader):
            img_q = img[0].cuda(rank, non_blocking=True)
            img_k = img[1].cuda(rank, non_blocking=True)

            out, label = model(img_q, img_k)
            loss = criterion(out, label)

            top1, top5 = accuracy(out, label, topk=(1, 5))
            if rank == 0:
                print(f"epoch:{epoch} batch:{batch} loss:{loss.item()} top1:{top1.item()}, top5:{top5.item()}")

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        if rank == 0:
            torch.save(model.module, f'MoCo-DDP-{epoch}.pth')


if __name__ == '__main__':
    world_size = torch.cuda.device_count()
    torch.multiprocessing.spawn(train, args=(world_size,), nprocs=world_size, join=True)
