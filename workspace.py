from typing import Self
import copy
from dataclasses import dataclass, field


@dataclass
class Tree:
    children: list[Self] = field(default_factory=list)


a = Tree()
b = Tree()
b.children.append(a)
c = Tree()
c.children.append(b)
d = Tree()
d.children.append(c)

node = d
history = [node]
while len(node.children) > 0:
    node = node.children[0]
    history.append(node)

history[3].children.append(Tree())

print(history)
