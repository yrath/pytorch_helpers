# -*- coding: utf-8 -*-

import torch

from collections import defaultdict


class NNTrainer(object):

    def __init__(self, model):
        self.model = model
        self.scheduler = None

        if torch.cuda.is_available():
            self.use_gpu = True

    def add_optimizer(self, optimizer="SGD", **kwargs):
        if isinstance(optimizer, str):
            optimizer = getattr(torch.optim, optimizer)
        self.optimizer = optimizer(self.model.parameters(), **kwargs)

    def add_lr_scheduler(self, scheduler="MultiplicativeLR", *args, **kwargs):
        if isinstance(scheduler, str):
            scheduler = getattr(torch.optim.lr_scheduler, scheduler)
        self.scheduler = scheduler(*args, **kwargs)

    def train(self, train_loader, valid_loader, n_epochs=10, output_key=None,
            metrics=None, optim_metric="loss"):
        if metrics is None or optim_metric not in metrics:
            raise ValueError("Missing required metric {} for network training, metrics: {}".format(
                optim_metric, metrics
            ))

        def run_epoch(data_loader, train=False):
            for metric in metrics.values():
                metric.reset()

            for inputs, weights, labels in data_loader:
                if self.use_gpu:
                    inputs, weights, labels = inputs.cuda(), weights.cuda(), labels.cuda()
                if train:
                    self.optimizer.zero_grad()

                outputs = self.model(inputs)
                if output_key is not None:
                    outputs = outputs.deep_get(output_key)
                # get last output of the NN
                output = outputs.deep_apply(lambda x: x[next(reversed(x))])

                for key, metric in metrics.items():
                    metric_value = metric.batch(inputs, output, weights, labels)
                    if train and key == optim_metric:
                        metric_value.backward()
                        self.optimizer.step()

            if self.scheduler is not None:
                self.scheduler.step()
            return {key: metric.aggregate() for key, metric in metrics.items()}

        results = defaultdict(list)
        for epoch in range(n_epochs):
            print("Starting epoch {}".format(epoch))

            results["train"].append(run_epoch(train_loader, train=True))
            results["valid"].append(run_epoch(valid_loader, train=False))

            print("Training loss: {}".format(results["train"][-1][optim_metric]))
            print("Validation loss: {}".format(results["valid"][-1][optim_metric]))

        return results
