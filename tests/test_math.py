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


def test_repr_is_data() -> None:
    assert repr(Scalar(2.5)) == "2.5"
    assert repr(Scalar(-3.0)) == "-3.0"


def test_add() -> None:
    assert (Scalar(2.0) + Scalar(3.5)).data == 5.5


def test_sub() -> None:
    assert (Scalar(10.0) - Scalar(4.25)).data == 5.75


def test_mul() -> None:
    assert (Scalar(3.0) * Scalar(2.0)).data == 6.0


def test_truediv() -> None:
    assert (Scalar(9.0) / Scalar(3.0)).data == 3.0


def test_pow() -> None:
    assert (Scalar(2.0) ** Scalar(3.0)).data == 8.0
