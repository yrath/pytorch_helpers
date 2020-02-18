# -*- coding: utf-8 -*-

import torch
import numpy as np


class Metric(object):

    def __init__(self):
        self.values = []

    def reset(self):
        self.values = []

    def batch(self, inputs, outputs, weights, labels):
        raise NotImplementedError()

    def aggregate(self):
        return np.mean(self.values)


class TorchLoss(Metric):

    def __init__(self, loss_func):
        super(TorchLoss, self).__init__()
        self.loss_func = loss_func

    def batch(self, inputs, outputs, weights, labels):
        loss = self.loss_func(outputs, labels)
        loss = torch.mean(weights * loss)
        self.values.append(loss.cpu().detach().numpy())
        return loss


class Accuracy(Metric):

    def batch(self, inputs, outputs, weights, labels):
        _, predictions = outputs.max(1)
        acc = ((labels == predictions) * weights).sum() / weights.sum()
        self.values.append(acc.cpu().detach().numpy())
        return acc


class ConfusionMatrix(Metric):

    def __init__(self, n_dim):
        self.n_dim = n_dim
        self.matrix = np.zeros((n_dim, n_dim))

    def reset(self):
        self.matrix = np.zeros((self.n_dim, self.n_dim))

    def batch(self, inputs, outputs, weights, labels):
        _, predictions = outputs.max(1)
        for pred, weight, label in zip(predictions.cpu(), weights.cpu(), labels.cpu()):
            self.matrix[pred, label] += 1
        return self.matrix

    def aggregate(self):
        for idx in range(self.n_dim):
            self.matrix[:, idx] /= np.sum(self.matrix[:, idx])
        return self.matrix
