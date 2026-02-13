import math
import operator
from collections.abc import Callable
from typing import Any

import pytest

from pygrad.math import (
    Add,
    BinaryOp,
    Div,
    Mul,
    Pow,
    Relu,
    Scalar,
    Sigmoid,
    Sub,
    Tanh,
    UnaryOp,
)


@pytest.mark.parametrize(
    ("op_cls", "left_data", "right_data", "expected_data"),
    [
        (Add, 2.0, 3.5, 5.5),
        (Sub, 10.0, 4.25, 5.75),
        (Mul, 3.0, 2.0, 6.0),
        (Div, 9.0, 3.0, 3.0),
        (Pow, 2.0, 3.0, 8.0),
    ],
    ids=["add", "sub", "mul", "div", "pow"],
)
def test_binary_op_out_is_expected_scalar(
    op_cls: type[Add | Sub | Mul | Div | Pow],
    left_data: float,
    right_data: float,
    expected_data: float,
) -> None:
    left = Scalar(left_data)
    right = Scalar(right_data)
    op = op_cls(left, right)

    actual = op.out

    assert isinstance(actual, Scalar)
    assert actual.data == pytest.approx(expected_data)
    assert actual.op is op


@pytest.mark.parametrize(
    ("op_cls", "operand_data", "expected_data"),
    [
        (Tanh, 0.5, 0.46211715726000974),
        (Sigmoid, -1.25, 0.22270013882530884),
        (Relu, -3.0, 0.0),
        (Relu, 2.5, 2.5),
    ],
    ids=["tanh", "sigmoid", "relu_negative", "relu_positive"],
)
def test_unary_op_out_is_expected_scalar(
    op_cls: type[Tanh | Sigmoid | Relu], operand_data: float, expected_data: float
) -> None:
    operand = Scalar(operand_data)
    op = op_cls(operand)

    actual = op.out

    assert isinstance(actual, Scalar)
    assert actual.data == pytest.approx(expected_data)
    assert actual.op is op


@pytest.mark.parametrize(
    ("binary_op", "left_data", "right_data", "op_cls", "expected_data"),
    [
        (operator.add, 2.0, 3.5, Add, 5.5),
        (operator.sub, 10.0, 4.25, Sub, 5.75),
        (operator.mul, 3.0, 2.0, Mul, 6.0),
        (operator.truediv, 9.0, 3.0, Div, 3.0),
        (operator.pow, 2.0, 3.0, Pow, 8.0),
    ],
    ids=["add", "sub", "mul", "truediv", "pow"],
)
def test_scalar_binary_dunders_return_expected_scalar(
    binary_op: Callable[[Scalar, Scalar], Scalar],
    left_data: float,
    right_data: float,
    op_cls: type[BinaryOp],
    expected_data: float,
) -> None:
    left = Scalar(left_data)
    right = Scalar(right_data)

    actual = binary_op(left, right)

    assert isinstance(actual, Scalar)
    assert actual.data == pytest.approx(expected_data)
    assert isinstance(actual.op, op_cls)
    assert actual.op.left is left
    assert actual.op.right is right


@pytest.mark.parametrize(
    ("method_name", "operand_data", "op_cls", "expected_data"),
    [
        ("tanh", 0.5, Tanh, 0.46211715726000974),
        ("sigmoid", -1.25, Sigmoid, 0.22270013882530884),
        ("relu", 2.5, Relu, 2.5),
    ],
    ids=["tanh", "sigmoid", "relu"],
)
def test_scalar_unary_methods_return_expected_scalar(
    method_name: str,
    operand_data: float,
    op_cls: type[UnaryOp],
    expected_data: float,
) -> None:
    operand = Scalar(operand_data)

    actual = getattr(operand, method_name)()

    assert isinstance(actual, Scalar)
    assert actual.data == pytest.approx(expected_data)
    assert isinstance(actual.op, op_cls)
    assert actual.op.operand is operand


@pytest.mark.parametrize(
    ("dunder_method", "other"),
    [
        (Scalar.__add__, 1),
        (Scalar.__sub__, 1),
        (Scalar.__mul__, 1),
        (Scalar.__truediv__, 1),
        (Scalar.__pow__, 1),
    ],
    ids=["add", "sub", "mul", "truediv", "pow"],
)
def test_scalar_binary_dunders_return_notimplemented_for_non_scalar(
    dunder_method: Callable[[Scalar, Any], object],
    other: Any,
) -> None:
    value = Scalar(2.0)

    actual = dunder_method(value, other)

    assert actual is NotImplemented


@pytest.mark.parametrize(
    "binary_op",
    [operator.add, operator.sub, operator.mul, operator.truediv, operator.pow],
    ids=["add", "sub", "mul", "truediv", "pow"],
)
def test_scalar_binary_operators_raise_type_error_for_non_scalar(
    binary_op: Callable[[Scalar, Any], Any],
) -> None:
    with pytest.raises(TypeError):
        binary_op(Scalar(2.0), 1)


def test_scalar_truediv_by_zero_raises_zero_division_error() -> None:
    with pytest.raises(ZeroDivisionError):
        _ = Scalar(1.0) / Scalar(0.0)


def test_scalar_repr_returns_data_string() -> None:
    assert repr(Scalar(3.25)) == "3.25"


@pytest.mark.parametrize(
    ("op_cls", "left_data", "right_data", "out_grad", "left_delta", "right_delta"),
    [
        (Add, 2.0, 3.5, 2.5, 2.5, 2.5),
        (Sub, 10.0, 4.25, 1.5, 1.5, -1.5),
        (Mul, 3.0, 2.0, 2.0, 4.0, 6.0),
        (Div, 9.0, 3.0, 1.5, 0.5, -1.5),
        (
            Pow,
            2.0,
            3.0,
            0.5,
            3.0 * (2.0**2.0) * 0.5,
            (2.0**3.0) * math.log(2.0) * 0.5,
        ),
    ],
    ids=["add", "sub", "mul", "div", "pow"],
)
def test_binary_op_backward_accumulates_expected_gradients(
    op_cls: type[Add | Sub | Mul | Div | Pow],
    left_data: float,
    right_data: float,
    out_grad: float,
    left_delta: float,
    right_delta: float,
) -> None:
    left = Scalar(left_data)
    right = Scalar(right_data)
    op = op_cls(left, right)
    out = op.out
    left.grad = 1.25
    right.grad = -0.75
    out.grad = out_grad

    op.backward()

    assert left.grad == pytest.approx(1.25 + left_delta)
    assert right.grad == pytest.approx(-0.75 + right_delta)


@pytest.mark.parametrize(
    ("op_cls", "operand_data", "out_grad", "delta"),
    [
        (Tanh, 0.5, 1.75, 1.376283532690373),
        (Sigmoid, -1.25, 2.0, 0.34620957398499414),
        (Relu, 2.5, 3.0, 3.0),
        (Relu, -2.5, 3.0, 0.0),
        (Relu, 0.0, 3.0, 0.0),
    ],
    ids=["tanh", "sigmoid", "relu_positive", "relu_negative", "relu_zero"],
)
def test_unary_op_backward_accumulates_expected_gradient(
    op_cls: type[Tanh | Sigmoid | Relu],
    operand_data: float,
    out_grad: float,
    delta: float,
) -> None:
    operand = Scalar(operand_data)
    op = op_cls(operand)
    out = op.out
    operand.grad = -0.5
    out.grad = out_grad

    op.backward()

    assert operand.grad == pytest.approx(-0.5 + delta)


def test_pow_backward_raises_value_error_for_non_positive_base() -> None:
    left = Scalar(0.0)
    right = Scalar(2.0)
    op = Pow(left, right)
    out = op.out
    out.grad = 1.0

    with pytest.raises(ValueError):
        op.backward()


def test_scalar_backward_resets_existing_graph_gradients() -> None:
    left = Scalar(2.0)
    right = Scalar(3.0)
    out = (left * right) + left
    left.grad = 9.0
    right.grad = -4.0
    out.grad = 12.0

    out.backward()

    assert out.grad == pytest.approx(1.0)
    assert left.grad == pytest.approx(4.0)
    assert right.grad == pytest.approx(2.0)


def test_scalar_backward_does_not_accumulate_across_calls() -> None:
    left = Scalar(2.0)
    right = Scalar(3.0)
    out = left * right

    out.backward()
    first_left_grad = left.grad
    first_right_grad = right.grad

    out.backward()

    assert left.grad == pytest.approx(first_left_grad)
    assert right.grad == pytest.approx(first_right_grad)
