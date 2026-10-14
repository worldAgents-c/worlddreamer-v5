# Copyright (C) 2026 World Agents Corporation.
import torch
import torch.distributed as dist
from lightning import LightningDataModule
from torch.utils.data import DataLoader, Sampler
from worlddreamer_v5.data.dataset import WorldDataset
from worlddreamer_v5.data.collator import WorldCollator


class TaskBalancedDistributedSampler(Sampler):
    def __init__(self, dataset, samples_per_task=4200, seed=42):
        self.dataset = dataset
        self.samples_per_task = int(samples_per_task)
        self.seed = int(seed)
        self.epoch = 0
        self.rank = (
            dist.get_rank() if dist.is_available() and dist.is_initialized() else 0
        )
        self.world_size = (
            dist.get_world_size()
            if dist.is_available() and dist.is_initialized()
            else 1
        )
        self.total_size = self.samples_per_task * len(dataset.task_indices)
        if self.total_size % self.world_size:
            raise ValueError(
                f"balanced epoch size {self.total_size} is not divisible by world size {self.world_size}"
            )
        self.num_samples = self.total_size // self.world_size

    def __iter__(self):
        generator = torch.Generator().manual_seed(self.seed + self.epoch)
        balanced = []
        for task in sorted(self.dataset.task_indices):
            source = torch.tensor(self.dataset.task_indices[task], dtype=torch.long)
            count = source.numel()
            full_repeats, remainder = divmod(self.samples_per_task, count)
            for _ in range(full_repeats):
                balanced.append(source[torch.randperm(count, generator=generator)])
            if remainder:
                balanced.append(
                    source[torch.randperm(count, generator=generator)[:remainder]]
                )
        indices = torch.cat(balanced)
        indices = indices[torch.randperm(indices.numel(), generator=generator)]
        indices = indices[self.rank : self.total_size : self.world_size]
        return iter(indices.tolist())

    def __len__(self):
        return self.num_samples

    def set_epoch(self, epoch):
        self.epoch = int(epoch)


class WorldDataModule(LightningDataModule):
    def __init__(self, params):
        super().__init__()
        self.params = dict(params)
        if int(params.get('batch_size', 1)) != 1:
            raise ValueError('per-device scene batch size must be one')
        self.collate_fn = WorldCollator(params['processor_path'])
        self.train_set = None

    def setup(self, stage=None):
        if self.train_set is None:
            self.train_set = WorldDataset(self.params)

    def train_dataloader(self):
        self.setup()
        workers = int(self.params.get('num_workers', 2))
        return DataLoader(self.train_set, batch_size=1,
            sampler=TaskBalancedDistributedSampler(self.train_set,
                self.params.get('samples_per_task', 4200), self.params.get('seed', 42)),
            num_workers=workers, collate_fn=self.collate_fn, pin_memory=True,
            persistent_workers=workers > 0)
