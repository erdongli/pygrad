from enum import StrEnum


class Op(StrEnum):
    ADD = "+"
    SUB = "-"
    MUL = "*"
    DIV = "/"
    POW = "**"


class Scalar:
    __slots__ = ("data", "op", "inputs")

    def __init__(
        self, data: float, op: Op | None = None, inputs: tuple["Scalar", ...] = ()
    ):
        self.data = data
        self.op = op
        self.inputs = inputs

    def __repr__(self) -> str:
        return f"{self.data}"

    def __add__(self, other: "Scalar") -> "Scalar":
        if not isinstance(other, Scalar):
            return NotImplemented
        return Scalar(self.data + other.data, op=Op.ADD, inputs=(self, other))

    def __sub__(self, other: "Scalar") -> "Scalar":
        if not isinstance(other, Scalar):
            return NotImplemented
        return Scalar(self.data - other.data, op=Op.SUB, inputs=(self, other))

    def __mul__(self, other: "Scalar") -> "Scalar":
        if not isinstance(other, Scalar):
            return NotImplemented
        return Scalar(self.data * other.data, op=Op.MUL, inputs=(self, other))

    def __truediv__(self, other: "Scalar") -> "Scalar":
        if not isinstance(other, Scalar):
            return NotImplemented
        return Scalar(self.data / other.data, op=Op.DIV, inputs=(self, other))

    def __pow__(self, other: "Scalar") -> "Scalar":
        if not isinstance(other, Scalar):
            return NotImplemented
        return Scalar(self.data**other.data, op=Op.POW, inputs=(self, other))
