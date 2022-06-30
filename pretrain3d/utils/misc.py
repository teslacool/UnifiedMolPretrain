import math
import torch
from torch_geometric.nn import global_mean_pool


class WarmCosine:
    def __init__(self, warmup=4e3, tmax=1e5, eta_min=5e-4):
        if warmup is None:
            self.warmup = 0
        else:
            warmup_step = int(warmup)
            assert warmup_step > 0
            self.warmup = warmup_step
            self.lr_step = (1 - eta_min) / warmup_step
        self.tmax = int(tmax)
        self.eta_min = eta_min

    def step(self, step):
        if step >= self.warmup:
            return (
                self.eta_min
                + (1 - self.eta_min)
                * (1 + math.cos(math.pi * (step - self.warmup) / self.tmax))
                / 2
            )

        else:
            return self.eta_min + self.lr_step * step


class WarmLinear:
    def __init__(self, warmup=4e3, tmax=1e5, eta_min=5e-4):
        if warmup is None:
            self.warmup_step = 0
        else:
            warmup_step = int(warmup)
            assert warmup_step > 0
            self.warmup_step = warmup_step
            self.warmup_lr_step = (1 - eta_min) / warmup_step
        self.decay_lr_step = (eta_min - 1) / (tmax - self.warmup_step)
        self.eta_min = eta_min

    def step(self, step):
        if step >= self.warmup_step:
            return max(self.eta_min, 1 + self.decay_lr_step * (step - self.warmup_step))
        else:
            return max(self.eta_min, self.eta_min + self.warmup_lr_step * step)


def get_random_rotation_3d(pos):
    random_quaternions = torch.randn(4).to(pos)
    random_quaternions = random_quaternions / random_quaternions.norm(dim=-1, keepdim=True)
    return torch.einsum("kj,ij->ki", pos, quaternion_to_rotation_matrix(random_quaternions))


def quaternion_to_rotation_matrix(quaternion):
    q0 = quaternion[0]
    q1 = quaternion[1]
    q2 = quaternion[2]
    q3 = quaternion[3]
    r00 = 2 * (q0 * q0 + q1 * q1) - 1
    r01 = 2 * (q1 * q2 - q0 * q3)
    r02 = 2 * (q1 * q3 + q0 * q2)
    r10 = 2 * (q1 * q2 + q0 * q3)
    r11 = 2 * (q0 * q0 + q2 * q2) - 1
    r12 = 2 * (q2 * q3 - q0 * q1)
    r20 = 2 * (q1 * q3 - q0 * q2)
    r21 = 2 * (q2 * q3 + q0 * q1)
    r22 = 2 * (q0 * q0 + q3 * q3) - 1
    return torch.stack([r00, r01, r02, r10, r11, r12, r20, r21, r22], dim=-1).reshape(3, 3)


class PreprocessBatch:
    def __init__(self, norm2origin=False, random_rotation=False) -> None:
        self.norm2origin = norm2origin
        self.random_rotation = random_rotation

    def process(self, batch):
        if not self.norm2origin and not self.random_rotation:
            return
        pos = batch.pos
        if self.norm2origin:
            pos_mean = global_mean_pool(pos, batch.batch)
            pos = pos - torch.repeat_interleave(pos_mean, batch.n_nodes, dim=0)
        if self.random_rotation:
            pos = get_random_rotation_3d(pos)
        batch.pos = pos
