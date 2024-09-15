import queue

import torch
import torch.nn as nn
import torchvision
from oauthlib.uri_validate import query
from torch import no_grad


class MoCo(nn.Module):
    def __init__(self, dim, momentum, T, Q):
        super(MoCo, self).__init__()
        self.encoder_q = torchvision.models.resnet50(num_classes=dim)
        self.encoder_k = torchvision.models.resnet50(num_classes=dim)
        # queue is trans F*Q
        self.queue = torch.randn(dim, Q).cuda()
        # 非常影响初始化
        self.queue = nn.functional.normalize(self.queue, dim=0)
        self.start = 0
        self.Q = Q
        self.momentum = momentum
        self.T = T
        for para in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            para[1].data = para[0].data
            para[1].requires_grad = False

    @torch.no_grad()
    def update_queue(self, k):
        k = concat_all_gather(k)
        print(self.start)
        batch_size = k.shape[0]
        assert self.start % batch_size == 0  # for simplicity
        self.queue[:, self.start:self.start + batch_size] = k.T
        self.start = (self.start + batch_size) % self.Q

    def momentum_update(self):
        for para in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            para[1].data = self.momentum * para[1].data + (1 - self.momentum) * para[0].data

    @torch.no_grad()
    def batch_shuffle_ddp(self, x):
        batch_size_this = x.shape[0]
        x_gather = concat_all_gather(x)
        batch_size_all = x_gather.shape[0]

        num_gpus = batch_size_all // batch_size_this

        # 生成随机排布作为当前位置经过shuffule后的数据
        idx_shuffle = torch.randperm(batch_size_all).cuda()

        # broadcast to all gpus
        torch.distributed.broadcast(idx_shuffle, src=0)

        # 根据新索引排序index，就能找到原来的值去了哪
        idx_unshuffle = torch.argsort(idx_shuffle)

        # 根据排布重新分配值
        gpu_idx = torch.distributed.get_rank()
        idx_this = idx_shuffle.view(num_gpus, -1)[gpu_idx]

        return x_gather[idx_this], idx_unshuffle

    @torch.no_grad()
    def batch_unshuffle_ddp(self, x, idx_unshuffle):
        batch_size_this = x.shape[0]
        x_gather = concat_all_gather(x)
        batch_size_all = x_gather.shape[0]

        # 从所有被shuffule的数据上恢复
        num_gpus = batch_size_all // batch_size_this
        gpu_idx = torch.distributed.get_rank()
        # 获取当前gpu原有的数据
        idx_this = idx_unshuffle.view(num_gpus, -1)[gpu_idx]

        return x_gather[idx_this]

    def forward(self, im_q, im_k):
        q = self.encoder_q(im_q)
        q = nn.functional.normalize(q, dim=-1)

        with torch.no_grad():
            self.momentum_update()
            im_k, shuffle_idx = self.batch_shuffle_ddp(im_k)
            k = self.encoder_k(im_k)
            k = nn.functional.normalize(k, dim=-1)
            k = self.batch_unshuffle_ddp(k, shuffle_idx)

        # 一开始直接用的torch.mm和torch.bmm
        # N here means batch size
        # nf,nf->n
        # N*1 q dot k
        pos_sim = torch.einsum("nf,nf->n", q, k).unsqueeze(-1)
        # N*Q q dot queue
        # 计算了梯度这里
        neg_sim = torch.einsum("nf,fk->nk", q, self.queue.clone().detach())
        # N*(1+Q) Each sample's pos sim and neg sim
        # logits is the similarity of the q k and bank
        logits = torch.cat([pos_sim, neg_sim], dim=1)
        logits /= self.T
        # pos 0 is k and the pos of similarity that should close to 1, others kept to zero
        # 第一次用的默认，报错了。改成long
        label = torch.zeros(logits.shape[0], dtype=torch.long).cuda()

        self.update_queue(k)

        return logits, label


@torch.no_grad()
def concat_all_gather(tensor):
    # 创建gpu_num个batch的数组
    tensors_gather = [
        torch.ones_like(tensor) for _ in range(torch.distributed.get_world_size())
    ]
    # 获取数据
    torch.distributed.all_gather(tensors_gather, tensor, async_op=False)

    output = torch.cat(tensors_gather, dim=0)
    return output
