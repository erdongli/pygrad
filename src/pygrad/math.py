import math
from abc import ABC, abstractmethod
from typing import Callable, ClassVar, override


def _sigmoid(x: float) -> float:
    if x >= 0.0:
        z = math.exp(-x)
        return 1.0 / (1.0 + z)
    z = math.exp(x)
    return z / (1.0 + z)


class Op(ABC):
    symbol: ClassVar[str] = "?"

    out: "Scalar"

    @abstractmethod
    def backward(self) -> None:
        pass

    @property
    @abstractmethod
    def operands(self) -> tuple["Scalar", ...]:
        pass


class NoOp(Op):
    symbol: ClassVar[str] = "noop"

    @override
    def backward(self) -> None:
        pass

    @property
    @override
    def operands(self) -> tuple["Scalar", ...]:
        return ()


NO_OP = NoOp()


class BinaryOp(Op):
    def __init__(
        self,
        left: "Scalar",
        right: "Scalar",
        forward_fn: Callable[[float, float], float],
    ) -> None:
        self.left = left
        self.right = right
        self.out = Scalar(forward_fn(left.data, right.data), op=self)

    @property
    @override
    def operands(self) -> tuple["Scalar", ...]:
        return (self.left, self.right)


class UnaryOp(Op):
    def __init__(self, operand: "Scalar", forward_fn: Callable[[float], float]) -> None:
        self.operand = operand
        self.out = Scalar(forward_fn(operand.data), op=self)

    @property
    @override
    def operands(self) -> tuple["Scalar", ...]:
        return (self.operand,)


class Add(BinaryOp):
    symbol: ClassVar[str] = "+"

    def __init__(self, left: "Scalar", right: "Scalar") -> None:
        super().__init__(left, right, lambda x, y: x + y)

    @override
    def backward(self) -> None:
        self.left.grad += 1.0 * self.out.grad
        self.right.grad += 1.0 * self.out.grad


class Sub(BinaryOp):
    symbol: ClassVar[str] = "-"

    def __init__(self, left: "Scalar", right: "Scalar") -> None:
        super().__init__(left, right, lambda x, y: x - y)

    @override
    def backward(self) -> None:
        self.left.grad += 1.0 * self.out.grad
        self.right.grad -= 1.0 * self.out.grad


class Mul(BinaryOp):
    symbol: ClassVar[str] = "*"

    def __init__(self, left: "Scalar", right: "Scalar") -> None:
        super().__init__(left, right, lambda x, y: x * y)

    @override
    def backward(self) -> None:
        self.left.grad += self.right.data * self.out.grad
        self.right.grad += self.left.data * self.out.grad


class Div(BinaryOp):
    symbol: ClassVar[str] = "/"

    def __init__(self, left: "Scalar", right: "Scalar") -> None:
        super().__init__(left, right, lambda x, y: x / y)

    @override
    def backward(self) -> None:
        self.left.grad += (1.0 / self.right.data) * self.out.grad
        self.right.grad += (-self.left.data / (self.right.data**2)) * self.out.grad


class Pow(BinaryOp):
    symbol: ClassVar[str] = "**"

    def __init__(self, left: "Scalar", right: "Scalar") -> None:
        super().__init__(left, right, lambda x, y: x**y)

    @override
    def backward(self) -> None:
        if self.left.data <= 0.0:
            raise ValueError(
                "Pow.backward requires a positive base for exponent gradients."
            )

        self.left.grad += (
            self.right.data
            * (self.left.data ** (self.right.data - 1.0))
            * self.out.grad
        )
        self.right.grad += self.out.data * math.log(self.left.data) * self.out.grad


class Tanh(UnaryOp):
    symbol: ClassVar[str] = "tanh"

    def __init__(self, operand: "Scalar") -> None:
        super().__init__(operand, lambda x: math.tanh(x))

    @override
    def backward(self) -> None:
        self.operand.grad += (1 - (self.out.data**2)) * self.out.grad


class Sigmoid(UnaryOp):
    symbol: ClassVar[str] = "sigmoid"

    def __init__(self, operand: "Scalar") -> None:
        super().__init__(operand, _sigmoid)

    @override
    def backward(self) -> None:
        self.operand.grad += self.out.data * (1 - self.out.data) * self.out.grad


class Relu(UnaryOp):
    symbol: ClassVar[str] = "relu"

    def __init__(self, operand: "Scalar") -> None:
        super().__init__(operand, lambda x: x if x > 0 else 0.0)

    @override
    def backward(self) -> None:
        if self.operand.data > 0:
            self.operand.grad += self.out.grad


class Scalar:
    __slots__ = ("data", "op", "grad")

    def __init__(self, data: float, op: Op = NO_OP) -> None:
        self.data = data
        self.op = op
        self.grad = 0.0

    def __repr__(self) -> str:
        return f"{self.data}"

    def __add__(self, other: "Scalar") -> "Scalar":
        if not isinstance(other, Scalar):
            return NotImplemented
        return Add(self, other).out

    def __sub__(self, other: "Scalar") -> "Scalar":
        if not isinstance(other, Scalar):
            return NotImplemented
        return Sub(self, other).out

    def __mul__(self, other: "Scalar") -> "Scalar":
        if not isinstance(other, Scalar):
            return NotImplemented
        return Mul(self, other).out

    def __truediv__(self, other: "Scalar") -> "Scalar":
        if not isinstance(other, Scalar):
            return NotImplemented
        return Div(self, other).out

    def __pow__(self, other: "Scalar") -> "Scalar":
        if not isinstance(other, Scalar):
            return NotImplemented
        return Pow(self, other).out

    def tanh(self) -> "Scalar":
        return Tanh(self).out

    def sigmoid(self) -> "Scalar":
        return Sigmoid(self).out

    def relu(self) -> "Scalar":
        return Relu(self).out

    def backward(self) -> None:
        topo, visited, stack = [], set(), [(self, False)]

        while stack:
            node, expanded = stack.pop()
            if expanded:
                topo.append(node)
                continue

            if node in visited:
                continue

            node.grad = 0.0
            visited.add(node)
            stack.append((node, True))

            for operand in reversed(node.op.operands):
                if operand not in visited:
                    stack.append((operand, False))

        self.grad = 1.0
        for node in reversed(topo):
            node.op.backward()
