# -*- coding: utf-8 -*-

import torch
from torch.utils import data


def create_dataloader(*arrays, **kwargs):
    tensors = [torch.as_tensor(array) for array in arrays]

    dataset = data.TensorDataset(*tensors)
    dataloader = data.DataLoader(dataset, **kwargs)
    return dataset, dataloader
