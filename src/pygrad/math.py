from abc import ABC, abstractmethod
from typing import ClassVar, override


class Op(ABC):
    symbol: ClassVar[str] = "?"

    @abstractmethod
    def forward(self) -> "Scalar":
        pass

    @abstractmethod
    def backward(self, out: "Scalar") -> None:
        pass

    @property
    @abstractmethod
    def operands(self) -> tuple["Scalar", ...]:
        pass


class BinaryOp(Op):
    def __init__(self, left: "Scalar", right: "Scalar") -> None:
        self.left = left
        self.right = right

    @property
    @override
    def operands(self) -> tuple["Scalar", ...]:
        return (self.left, self.right)


class Add(BinaryOp):
    symbol: ClassVar[str] = "+"

    @override
    def forward(self) -> "Scalar":
        return Scalar(self.left.data + self.right.data, op=self)

    @override
    def backward(self, out: "Scalar") -> None:
        raise NotImplementedError("Add.backward is not implemented yet.")


class Sub(BinaryOp):
    symbol: ClassVar[str] = "-"

    @override
    def forward(self) -> "Scalar":
        return Scalar(self.left.data - self.right.data, op=self)

    @override
    def backward(self, out: "Scalar") -> None:
        raise NotImplementedError("Sub.backward is not implemented yet.")


class Mul(BinaryOp):
    symbol: ClassVar[str] = "*"

    @override
    def forward(self) -> "Scalar":
        return Scalar(self.left.data * self.right.data, op=self)

    @override
    def backward(self, out: "Scalar") -> None:
        raise NotImplementedError("Mul.backward is not implemented yet.")


class Div(BinaryOp):
    symbol: ClassVar[str] = "/"

    @override
    def forward(self) -> "Scalar":
        return Scalar(self.left.data / self.right.data, op=self)

    @override
    def backward(self, out: "Scalar") -> None:
        raise NotImplementedError("Div.backward is not implemented yet.")


class Pow(BinaryOp):
    symbol: ClassVar[str] = "**"

    @override
    def forward(self) -> "Scalar":
        return Scalar(self.left.data**self.right.data, op=self)

    @override
    def backward(self, out: "Scalar") -> None:
        raise NotImplementedError("Pow.backward is not implemented yet.")


class Scalar:
    __slots__ = ("data", "op")

    def __init__(self, data: float, op: Op | None = None) -> None:
        self.data = data
        self.op = op

    def __repr__(self) -> str:
        return f"{self.data}"

    def __add__(self, other: "Scalar") -> "Scalar":
        if not isinstance(other, Scalar):
            return NotImplemented
        return Add(self, other).forward()

    def __sub__(self, other: "Scalar") -> "Scalar":
        if not isinstance(other, Scalar):
            return NotImplemented
        return Sub(self, other).forward()

    def __mul__(self, other: "Scalar") -> "Scalar":
        if not isinstance(other, Scalar):
            return NotImplemented
        return Mul(self, other).forward()

    def __truediv__(self, other: "Scalar") -> "Scalar":
        if not isinstance(other, Scalar):
            return NotImplemented
        return Div(self, other).forward()

    def __pow__(self, other: "Scalar") -> "Scalar":
        if not isinstance(other, Scalar):
            return NotImplemented
        return Pow(self, other).forward()
