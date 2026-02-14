import math

import pytest

from pygrad.math import Scalar
from pygrad.nn import Layer, MLP, Neuron


def test_neuron_init_samples_weights_and_bias(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sampled_values = iter([0.1, -0.2, 0.3])
    calls: list[tuple[int, int]] = []

    def fake_uniform(low: int, high: int) -> float:
        calls.append((low, high))
        return next(sampled_values)

    monkeypatch.setattr("pygrad.nn.random.uniform", fake_uniform)

    expect_weights = [0.1, -0.2]
    expect_bias = 0.3
    actual = Neuron(2)

    assert calls == [(-1, 1), (-1, 1), (-1, 1)]
    assert [w.data for w in actual.weights] == pytest.approx(expect_weights)
    assert actual.bias.data == pytest.approx(expect_bias)


def test_neuron_call_returns_tanh_of_affine_combination() -> None:
    neuron = Neuron(2)
    neuron.weights = [Scalar(0.5), Scalar(-1.0)]
    neuron.bias = Scalar(0.25)

    expect = math.tanh(4.25)
    actual = neuron([Scalar(2.0), Scalar(-3.0)])

    assert actual.data == pytest.approx(expect)


def test_neuron_call_raises_for_input_shape_mismatch() -> None:
    neuron = Neuron(2)

    with pytest.raises(
        ValueError,
        match=r"^input shape mismatch: expected 2, got 1$",
    ):
        neuron([Scalar(2.0)])


def test_layer_call_applies_each_neuron_to_input() -> None:
    layer = Layer(2, 3)
    layer.neurons[0].weights = [Scalar(1.0), Scalar(0.0)]
    layer.neurons[0].bias = Scalar(0.0)
    layer.neurons[1].weights = [Scalar(0.0), Scalar(1.0)]
    layer.neurons[1].bias = Scalar(0.0)
    layer.neurons[2].weights = [Scalar(1.0), Scalar(1.0)]
    layer.neurons[2].bias = Scalar(-1.0)

    expect = [math.tanh(0.5), math.tanh(-0.25), math.tanh(-0.75)]
    actual = layer([Scalar(0.5), Scalar(-0.25)])

    assert len(actual) == 3
    assert [v.data for v in actual] == pytest.approx(expect)


def test_mlp_init_expected_layer_shapes() -> None:
    actual = MLP([2, 3, 1])

    assert len(actual.layers) == 2
    assert len(actual.layers[0].neurons) == 3
    assert len(actual.layers[1].neurons) == 1
    assert all(len(n.weights) == 2 for n in actual.layers[0].neurons)
    assert all(len(n.weights) == 3 for n in actual.layers[1].neurons)


def test_mlp_backward_propogates_expected_input_gradients() -> None:
    mlp = MLP([2, 2, 1])
    mlp.layers[0].neurons[0].weights = [Scalar(1.0), Scalar(0.0)]
    mlp.layers[0].neurons[0].bias = Scalar(0.0)
    mlp.layers[0].neurons[1].weights = [Scalar(0.0), Scalar(1.0)]
    mlp.layers[0].neurons[1].bias = Scalar(0.0)
    mlp.layers[1].neurons[0].weights = [Scalar(1.0), Scalar(1.0)]
    mlp.layers[1].neurons[0].bias = Scalar(0.0)

    x0 = Scalar(0.2)
    x1 = Scalar(-0.3)

    tanh_x0 = math.tanh(0.2)
    tanh_x1 = math.tanh(-0.3)
    tanh_sum = math.tanh(tanh_x0 + tanh_x1)
    common = 1.0 - tanh_sum**2

    expect_x0_grad = common * (1.0 - tanh_x0**2)
    expect_x1_grad = common * (1.0 - tanh_x1**2)
    mlp([x0, x1])[0].backward()

    assert x0.grad == pytest.approx(expect_x0_grad)
    assert x1.grad == pytest.approx(expect_x1_grad)
