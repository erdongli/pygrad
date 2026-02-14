import random

from pygrad.math import Scalar


class Neuron:
    def __init__(self, n: int) -> None:
        self.weights = [Scalar(random.uniform(-1, 1)) for _ in range(n)]
        self.bias = Scalar(random.uniform(-1, 1))

    def __call__(self, x: list[Scalar]) -> Scalar:
        if len(x) != len(self.weights):
            raise ValueError(
                f"input shape mismatch: expected {len(self.weights)}, got {len(x)}"
            )
        z = sum((wi * xi for wi, xi in zip(self.weights, x)), self.bias)
        return z.tanh()

    def parameters(self) -> list[Scalar]:
        return self.weights + [self.bias]


class Layer:
    def __init__(self, n_prev: int, n: int) -> None:
        self.neurons = [Neuron(n_prev) for _ in range(n)]

    def __call__(self, x: list[Scalar]) -> list[Scalar]:
        return [n(x) for n in self.neurons]

    def parameters(self) -> list[Scalar]:
        return [p for n in self.neurons for p in n.parameters()]


class MLP:
    def __init__(self, ns: list[int]) -> None:
        self.layers = [Layer(ns[i - 1], ns[i]) for i in range(1, len(ns))]

    def __call__(self, x: list[Scalar]) -> list[Scalar]:
        for layer in self.layers:
            x = layer(x)
        return x

    def parameters(self) -> list[Scalar]:
        return [p for layer in self.layers for p in layer.parameters()]
