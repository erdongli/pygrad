from graphviz import Digraph

from pygrad.math import Scalar


def render(root: Scalar, filename: str) -> None:
    dot = Digraph(format="svg")
    dot.attr(rankdir="LR")

    queue, node_names = [root], set()
    while len(queue):
        node = queue.pop()
        node_name = str(id(node))
        if node_name in node_names:
            continue
        node_names.add(node_name)

        dot.node(node_name, label=str(node))

        if not node.op:
            continue
        op = node.op

        op_name = node_name + op.symbol
        dot.node(op_name, label=op.symbol)
        dot.edge(op_name, node_name)
        for operand in op.operands:
            queue.append(operand)
            dot.edge(str(id(operand)), op_name)

    dot.render(filename, cleanup=True)
