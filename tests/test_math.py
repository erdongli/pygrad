import pytest

from pygrad.math import Op, Scalar


def test_op_enum_values_are_str() -> None:
    assert isinstance(Op.ADD.value, str)
    assert isinstance(Op.SUB.value, str)
    assert isinstance(Op.MUL.value, str)
    assert isinstance(Op.DIV.value, str)
    assert isinstance(Op.POW.value, str)


def test_op_enum_members_exist() -> None:
    assert Op.ADD.name == "ADD"
    assert Op.SUB.name == "SUB"
    assert Op.MUL.name == "MUL"
    assert Op.DIV.name == "DIV"
    assert Op.POW.name == "POW"


def test_op_enum_strenum_auto_values_are_lowercase_names() -> None:
    assert Op.ADD.value == "+"
    assert Op.SUB.value == "-"
    assert Op.MUL.value == "*"
    assert Op.DIV.value == "/"
    assert Op.POW.value == "**"


def test_scalar_defaults_op_to_none() -> None:
    actual = Scalar(1.0)
    assert actual.op is None
    assert actual.inputs == ()


def test_repr_is_data() -> None:
    assert repr(Scalar(2.5)) == "2.5"
    assert repr(Scalar(-3.0)) == "-3.0"


def test_add() -> None:
    lhs = Scalar(2.0)
    rhs = Scalar(3.5)
    actual = lhs + rhs
    assert actual.data == 5.5
    assert actual.op == Op.ADD
    assert actual.inputs == (lhs, rhs)


def test_sub() -> None:
    lhs = Scalar(10.0)
    rhs = Scalar(4.25)
    actual = lhs - rhs
    assert actual.data == 5.75
    assert actual.op == Op.SUB
    assert actual.inputs == (lhs, rhs)


def test_mul() -> None:
    lhs = Scalar(3.0)
    rhs = Scalar(2.0)
    actual = lhs * rhs
    assert actual.data == 6.0
    assert actual.op == Op.MUL
    assert actual.inputs == (lhs, rhs)


def test_truediv() -> None:
    lhs = Scalar(9.0)
    rhs = Scalar(3.0)
    actual = lhs / rhs
    assert actual.data == 3.0
    assert actual.op == Op.DIV
    assert actual.inputs == (lhs, rhs)


def test_pow() -> None:
    lhs = Scalar(2.0)
    rhs = Scalar(3.0)
    actual = lhs**rhs
    assert actual.data == 8.0
    assert actual.op == Op.POW
    assert actual.inputs == (lhs, rhs)


@pytest.mark.parametrize(
    "operation",
    [
        lambda x: x + 2.0,
        lambda x: x - 2.0,
        lambda x: x * 2.0,
        lambda x: x / 2.0,
        lambda x: x**2.0,
    ],
)
def test_operators_with_non_scalar_rhs_raise_type_error(operation) -> None:
    with pytest.raises(TypeError):
        operation(Scalar(1.0))
