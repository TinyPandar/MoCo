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
        self.register_buffer("queue", torch.randn(dim, Q))
        self.start = 0
        self.Q = Q
        self.momentum = momentum
        self.T = T

    def update_queue(self, k):
        batch_size = k.shape[0]
        self.queue[:, self.start:self.start + batch_size] = k.T
        self.start = (self.start + batch_size) % self.Q

    def momentum_update(self):
        for para in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            para[1].data = self.momentum * para[1].data + (1 - self.momentum) * para[0].data

    def forward(self, im_q, im_k):
        q = self.encoder_q(im_q)
        q = nn.functional.normalize(q, dim=-1)

        with torch.no_grad():
            self.momentum_update()
            k = self.encoder_k(im_k)
            k = nn.functional.normalize(k, dim=-1)

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
        label = torch.zeros(logits.shape[0], dtype=torch.long)

        self.update_queue(k)

        return logits, label
