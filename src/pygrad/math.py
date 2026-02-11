from enum import StrEnum, auto


class Op(StrEnum):
    ADD = auto()
    SUB = auto()
    MUL = auto()
    DIV = auto()
    POW = auto()


class Scalar:
    def __init__(self, data: float, op: Op | None = None):
        self.data = data
        self.op = op

    def __repr__(self) -> str:
        return f"{self.data}"

    def __add__(self, other: "Scalar") -> "Scalar":
        if not isinstance(other, Scalar):
            return NotImplemented
        return Scalar(self.data + other.data, op=Op.ADD)

    def __sub__(self, other: "Scalar") -> "Scalar":
        if not isinstance(other, Scalar):
            return NotImplemented
        return Scalar(self.data - other.data, op=Op.SUB)

    def __mul__(self, other: "Scalar") -> "Scalar":
        if not isinstance(other, Scalar):
            return NotImplemented
        return Scalar(self.data * other.data, op=Op.MUL)

    def __truediv__(self, other: "Scalar") -> "Scalar":
        if not isinstance(other, Scalar):
            return NotImplemented
        return Scalar(self.data / other.data, op=Op.DIV)

    def __pow__(self, other: "Scalar") -> "Scalar":
        if not isinstance(other, Scalar):
            return NotImplemented
        return Scalar(self.data**other.data, op=Op.POW)
