import random

from pygrad.math import Scalar


class Neuron:
    def __init__(self, n: int) -> None:
        self.w = [Scalar(random.uniform(-1, 1)) for _ in range(n)]
        self.b = Scalar(random.uniform(-1, 1))

    def __call__(self, x: list[Scalar]) -> Scalar:
        if len(x) != len(self.w):
            raise ValueError(
                f"input shape mismatch: expected {len(self.w)}, got {len(x)}"
            )
        z = sum((wi * xi for wi, xi in zip(self.w, x)), self.b)
        return z.tanh()


class Layer:
    def __init__(self, n_prev: int, n: int) -> None:
        self.neurons = [Neuron(n_prev) for _ in range(n)]

    def __call__(self, x: list[Scalar]) -> list[Scalar]:
        return [n(x) for n in self.neurons]


class MLP:
    def __init__(self, ns: list[int]) -> None:
        self.layers = [Layer(ns[i - 1], ns[i]) for i in range(1, len(ns))]

    def __call__(self, x: list[Scalar]) -> list[Scalar]:
        for layer in self.layers:
            x = layer(x)
        return x
