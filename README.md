# pygrad

`pygrad` is a tiny educational reverse-mode autodiff engine in pure Python.
It is intentionally small so you can read the whole codebase and understand
how autodiff works under the hood.

## Install

```bash
uv sync
```

## Quickstart

```bash
uv run python - <<'PY'
from pygrad.math import Scalar

x = Scalar(2.0)
y = Scalar(-3.0)
z = (x * y + x**2).tanh()
z.backward()

print("z =", z.data)
print("dz/dx =", x.grad)
print("dz/dy =", y.grad)
PY
```

## MLP Example

```bash
uv run python - <<'PY'
from pygrad.math import Scalar
from pygrad.nn import MLP

mlp = MLP([2, 3, 1])
x = [Scalar(0.2), Scalar(-0.1)]
out = mlp(x)[0]
loss = (out - 1.0) ** 2
loss.backward()

lr = 0.05
for p in mlp.parameters():
    p.data -= lr * p.grad

print("loss =", loss.data)
PY
```

## Graph Rendering

Install Graphviz system binaries so `dot` is available, then:

```bash
uv run python - <<'PY'
from pygrad.math import Scalar
from pygrad.viz import render

a = Scalar(2.0)
b = Scalar(3.0)
c = (a * b + a).relu()
render(c, "graph")  # writes graph.svg
PY
```

## Test

```bash
uv run pytest
```

## License

MIT. See `LICENSE`.
