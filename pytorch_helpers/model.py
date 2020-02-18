# -*- coding: utf-8 -*-

from torch import nn

from collections import OrderedDict
from itertools import chain


class DeepOrderedDict(OrderedDict):
    def deep_get(self, keys):
        # get element of a nested dict from a list of keys
        if isinstance(keys, str):
            return self[keys]
        elif len[keys] == 1:
            return self[keys[0]]
        else:
            return self[keys[0]].deep_get(keys[1:])

    def deep_apply(self, func):
        # apply a function to the dictionary, repeating while a DeepOrderedDict is returned
        # example: deep_apply(lambda x: x[next(x)]) to get the first element in a nested dict
        output = self
        while True:
            output = func(output)
            if not isinstance(output, DeepOrderedDict):
                return output


class Module(nn.Module):

    def __init__(self, name, inputs=None):
        super(Module, self).__init__()
        self.name = name
        self.inputs = inputs
        self.transformations = OrderedDict()

    def add_transformation(self, transformation, name):
        if name in self.transformations:
            raise Exception("Duplicate transformation name {} in module {}.".format(name, self.name))
        self.transformations[name] = transformation

    def add_torch_layer(self, name=None, inputs=None, module=None, activation=None):
        # if no layer name is given, use 'layer_<#transformations>'
        if name is None:
            name = "layer_{}".format(len(self.transformations))
        # by default, new layer takes preceeding transformation as input
        if inputs is None:
            inputs = [next(reversed(self.transformations))]

        layer = Module(name, inputs)
        if module is not None:
            layer.add_transformation(module, "module")
        if activation is not None:
            layer.add_transformation(activation, "activation")
        self.add_transformation(layer, name)

    def forward(self, *x):
        outputs = DeepOrderedDict()
        outputs["input"] = x[0] if len(x) == 1 else x
        for name, transformation in self.transformations.items():
            if isinstance(transformation, Module):
                inputs = [outputs.deep_get(inp) for inp in transformation.inputs]
            else:
                inputs = [outputs]
            # if any of the inputs is not fully determined (a DeepOrderedDict), default to taking the last element (nested)
            inputs = [inp.deep_apply(lambda var: var[next(reversed(var))]) if isinstance(inp, DeepOrderedDict) else inp for inp in inputs]
            out = transformation(*inputs)
            outputs[name] = out
        return outputs

    def parameters(self):
        return chain(*[transformation.parameters() for transformation in
            self.transformations.values() if hasattr(transformation, "parameters")])

    def cuda(self):
        for key, transformation in self.transformations.items():
            if hasattr(transformation, "cuda"):
                self.transformations[key] = transformation.cuda()
        return self
