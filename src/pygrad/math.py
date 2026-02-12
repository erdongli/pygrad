import math
from abc import ABC, abstractmethod
from typing import ClassVar, Final, override


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
        self.left: Final = left
        self.right: Final = right

    @property
    @override
    def operands(self) -> tuple["Scalar", ...]:
        return (self.left, self.right)


class UnaryOp(Op):
    def __init__(self, operand: "Scalar") -> None:
        self.operand: Final = operand

    @property
    @override
    def operands(self) -> tuple["Scalar", ...]:
        return (self.operand,)


class Add(BinaryOp):
    symbol: ClassVar[str] = "+"

    @override
    def forward(self) -> "Scalar":
        return Scalar(self.left.data + self.right.data, op=self)

    @override
    def backward(self, out: "Scalar") -> None:
        self.left.grad += 1.0 * out.grad
        self.right.grad += 1.0 * out.grad


class Sub(BinaryOp):
    symbol: ClassVar[str] = "-"

    @override
    def forward(self) -> "Scalar":
        return Scalar(self.left.data - self.right.data, op=self)

    @override
    def backward(self, out: "Scalar") -> None:
        self.left.grad += 1.0 * out.grad
        self.right.grad -= 1.0 * out.grad


class Mul(BinaryOp):
    symbol: ClassVar[str] = "*"

    @override
    def forward(self) -> "Scalar":
        return Scalar(self.left.data * self.right.data, op=self)

    @override
    def backward(self, out: "Scalar") -> None:
        self.left.grad += self.right.data * out.grad
        self.right.grad += self.left.data * out.grad


class Div(BinaryOp):
    symbol: ClassVar[str] = "/"

    @override
    def forward(self) -> "Scalar":
        return Scalar(self.left.data / self.right.data, op=self)

    @override
    def backward(self, out: "Scalar") -> None:
        self.left.grad += (1.0 / self.right.data) * out.grad
        self.right.grad += (-self.left.data / (self.right.data**2)) * out.grad


class Pow(BinaryOp):
    symbol: ClassVar[str] = "**"

    @override
    def forward(self) -> "Scalar":
        return Scalar(self.left.data**self.right.data, op=self)

    @override
    def backward(self, out: "Scalar") -> None:
        if self.left.data <= 0.0:
            raise ValueError(
                "Pow.backward requires a positive base for exponent gradients."
            )

        self.left.grad += (
            self.right.data * (self.left.data ** (self.right.data - 1.0)) * out.grad
        )
        self.right.grad += out.data * math.log(self.left.data) * out.grad


class Tanh(UnaryOp):
    synbol: ClassVar[str] = "tanh"

    @override
    def forward(self) -> "Scalar":
        return Scalar(
            (math.exp(2 * self.operand.data) - 1)
            / (math.exp(2 * self.operand.data) + 1),
            op=self,
        )

    @override
    def backward(self, out: "Scalar") -> None:
        self.operand.grad += (1 - (self.forward().data ** 2)) * out.grad


class Scalar:
    __slots__ = ("data", "op", "grad")

    def __init__(self, data: float, op: Op | None = None) -> None:
        self.data = data
        self.op: Final = op
        self.grad = 0.0

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
