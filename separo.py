from dataclasses import dataclass, field
import numpy as np
from enum import Enum
from typing import Optional, NewType, NamedTuple
from collections import deque


@dataclass
class Coord:
    x: int
    y: int


Move = NewType("Move", tuple[Coord, Coord, Coord])


class Color(Enum):
    Red = 0
    Blue = 1


def opponent_of(color: Color) -> Color:
    match color:
        case Color.Red:
            return Color.Blue
        case Color.Blue:
            return Color.Red
        case _:
            raise ValueError(f"Unexpected color: {color}")


class NodePos(Enum):
    N: int = 0
    E: int = 1
    W: int = 2
    S: int = 3


@dataclass
class Node:
    region: Optional[int] = None
    edges: list[tuple[Coord, NodePos]] = field(default_factory=list)


@dataclass
class Graph:
    ngrids: int
    nodes: list[Node]

    def at(self, crd: Coord, pos: NodePos) -> Node:
        idx = (crd.x * self.ngrids + crd.y) * 4 + pos.value
        return self.nodes[idx]

    def __init__(self, width: int) -> None:
        # assert 3 < width < 20
        self.ngrids = width - 1
        self.nodes = [Node() for _ in range(self.ngrids * self.ngrids * 4)]

        x_max = self.ngrids - 1
        y_max = self.ngrids - 1

        for x in range(0, self.ngrids):
            for y in range(0, self.ngrids):
                crd = Coord(x, y)
                self.at(crd, NodePos.N).edges.append((crd, NodePos.E))
                self.at(crd, NodePos.N).edges.append((crd, NodePos.W))
                self.at(crd, NodePos.E).edges.append((crd, NodePos.N))
                self.at(crd, NodePos.E).edges.append((crd, NodePos.S))
                self.at(crd, NodePos.S).edges.append((crd, NodePos.E))
                self.at(crd, NodePos.S).edges.append((crd, NodePos.W))
                self.at(crd, NodePos.W).edges.append((crd, NodePos.N))
                self.at(crd, NodePos.W).edges.append((crd, NodePos.S))

                if x != 0:
                    self.at(crd, NodePos.W).edges.append((Coord(x - 1, y), NodePos.E))
                if x != x_max:
                    self.at(crd, NodePos.E).edges.append((Coord(x + 1, y), NodePos.W))
                if y != 0:
                    self.at(crd, NodePos.N).edges.append((Coord(x, y - 1), NodePos.S))
                if y != y_max:
                    self.at(crd, NodePos.S).edges.append((Coord(x, y + 1), NodePos.N))

    def remove_edge(self, crd1: Coord, pos1: NodePos, crd2: Coord, pos2: NodePos):
        self.at(crd1, pos1).edges.remove((crd2, pos2))
        self.at(crd2, pos2).edges.remove((crd1, pos1))

    def apply_move(self, next_move: Move):
        stone1, stone2, stone3 = next_move

        dx = stone2.x - stone1.x
        dy = stone2.y - stone1.y
        match (dx, dy):
            case (1, 1):
                self.remove_edge(stone1, NodePos.N, stone1, NodePos.W)
                self.remove_edge(stone1, NodePos.S, stone1, NodePos.E)
            case (1, -1):
                pos = Coord(stone1.x, stone1.y - 1)
                if 0 <= pos.y:
                    self.remove_edge(pos, NodePos.N, pos, NodePos.E)
                    self.remove_edge(pos, NodePos.S, pos, NodePos.W)
            case (-1, 1):
                pos = Coord(stone1.x - 1, stone1.y)
                if 0 <= pos.x:
                    self.remove_edge(pos, NodePos.N, pos, NodePos.E)
                    self.remove_edge(pos, NodePos.S, pos, NodePos.W)
            case (-1, -1):
                self.remove_edge(stone2, NodePos.N, stone2, NodePos.W)
                self.remove_edge(stone2, NodePos.S, stone2, NodePos.E)
            case _:
                raise ValueError(f"Invalid move: {next_move}")

        dx = stone3.x - stone2.x
        dy = stone3.y - stone2.y

        upper = self.ngrids
        match (dx, dy):
            case (1, 0):
                if stone2.x < upper and 1 <= stone2.y < upper:
                    self.remove_edge(
                        Coord(stone2.x, stone2.y - 1), NodePos.S, stone2, NodePos.N
                    )
            case (-1, 0):
                if stone3.x < upper and 1 <= stone3.y < upper:
                    self.remove_edge(
                        Coord(stone3.x, stone3.y - 1), NodePos.S, stone3, NodePos.N
                    )
            case (0, 1):
                if 1 <= stone2.x < upper and stone2.y < upper:
                    self.remove_edge(
                        Coord(stone2.x - 1, stone2.y), NodePos.E, stone2, NodePos.W
                    )
            case (0, -1):
                if 1 <= stone3.x < upper and stone3.y < upper:
                    self.remove_edge(
                        Coord(stone3.x - 1, stone3.y), NodePos.E, stone3, NodePos.W
                    )
            case _:
                raise ValueError(f"Invalud move: {next_move}")

    def find_connected_component(self, crd0: Coord, pos0: NodePos):
        region = self.at(crd0, pos0).region
        queue = deque()
        queue.append((crd0, pos0))
        size = 0
        while len(queue) > 0:
            crd, pos = queue.popleft()
            size += 1
            for n_crd, n_pos in self.at(crd, pos).edges:
                if self.at(n_crd, n_pos).region is None:
                    self.at(n_crd, n_pos).region = region
                    queue.append((n_crd, n_pos))
        return size

    def score(self) -> int:
        score: int = 0
        region: int = 0
        for idx in range(len(self.nodes)):
            if self.nodes[idx].region is not None:
                continue
            self.nodes[idx].region = region
            pos = NodePos(idx % 4)
            x = idx // 4 // self.ngrids
            y = idx // 4 % self.ngrids
            crd = Coord(x, y)
            if 4 < self.find_connected_component(crd, pos):
                score += 1
            region += 1
        return score


class Dir(NamedTuple):
    x: int
    y: int


@dataclass
class Grid:
    color: Optional[Color] = None
    roots: list[Dir] = field(default_factory=list)

    def is_valid_root(self, dir: Dir) -> bool:
        for d in self.roots:
            if abs(d[0] - dir[0]) + abs(d[1] - dir[1]) <= 1:
                return False
        return True


@dataclass
class Board:
    width: int
    grids: list[Grid]
    red_history: list[Move] = field(default_factory=list)
    blue_history: list[Move] = field(default_factory=list)

    def __init__(self, width: int):
        self.width = width
        self.grids = [Grid() for _ in range(width * width)]
        self.red_history = []
        self.blue_history = []

        lower = 0
        upper = width - 1
        self.grids[lower * width + lower].color = Color.Red
        self.grids[lower * width + upper].color = Color.Blue
        self.grids[upper * width + lower].color = Color.Blue
        self.grids[upper * width + upper].color = Color.Red

    def possible_moves(self, turn: Color) -> list[Move]:
        moves = []
        for idx1, grid in enumerate(self.grids):
            if grid.color != turn:
                continue
            x1 = idx1 // self.width
            y1 = idx1 % self.width
            for dir1 in [Dir(1, 1), Dir(-1, 1), Dir(-1, -1), Dir(1, -1)]:
                if not grid.is_valid_root(dir1):
                    continue
                x2 = x1 + dir1[0]
                y2 = y1 + dir1[1]
                if not (0 <= x2 < self.width and 0 <= y2 < self.width):
                    continue
                idx2 = x2 * self.width + y2
                if self.grids[idx2].color is not None:
                    continue
                if not self.grids[idx2].is_valid_root(Dir(-dir1[0], -dir1[1])):
                    continue
                for dir2 in [Dir(dir1[0], 0), Dir(0, dir1[1])]:
                    if not self.grids[idx2].is_valid_root(dir2):
                        continue
                    x3 = x2 + dir2[0]
                    y3 = y2 + dir2[1]
                    if not (0 <= x3 < self.width and 0 <= y3 < self.width):
                        continue
                    idx3 = x3 * self.width + y3
                    if self.grids[idx3].color != opponent_of(turn) and self.grids[
                        idx3
                    ].is_valid_root(Dir(-dir2[0], -dir2[1])):
                        moves.append(
                            Move((Coord(x1, y1), Coord(x2, y2), Coord(x3, y3)))
                        )
        return moves

    def is_valid_move(self, turn: Color, next_move: Move):
        return next_move in self.possible_moves(turn)

    def can_move(self, turn: Color):
        return len(self.possible_moves(turn)) > 0

    def is_gameover(self) -> bool:
        return (not self.can_move(Color.Red)) and (not self.can_move(Color.Blue))

    def apply_move(self, next_move: Move, turn: Color):
        stone1, stone2, stone3 = next_move
        idx1 = stone1.x * self.width + stone1.y
        idx2 = stone2.x * self.width + stone2.y
        idx3 = stone3.x * self.width + stone3.y

        self.grids[idx2].color = turn
        self.grids[idx3].color = turn

        self.grids[idx1].roots.append(Dir(stone2.x - stone1.x, stone2.y - stone1.y))
        self.grids[idx2].roots.append(Dir(stone1.x - stone2.x, stone1.y - stone2.y))
        self.grids[idx2].roots.append(Dir(stone3.x - stone2.x, stone3.y - stone2.y))
        self.grids[idx3].roots.append(Dir(stone2.x - stone3.x, stone2.y - stone3.y))

        match turn:
            case Color.Red:
                self.red_history.append(next_move)
            case Color.Blue:
                self.blue_history.append(next_move)

    def to_chars(self) -> list[list[str]]:
        s = [
            [" " for _ in range(self.width * 2 - 1)] for _ in range(self.width * 2 - 1)
        ]
        for i in range(self.width):
            for j in range(self.width):
                idx = i * self.width + j
                match self.grids[idx].color:
                    case Color.Red:
                        s[i * 2][j * 2] = "@"
                    case Color.Blue:
                        s[i * 2][j * 2] = "#"
                    case None:
                        s[i * 2][j * 2] = "."

                for d in self.grids[idx].roots:
                    ni = i * 2 + d[0]
                    nj = j * 2 + d[1]
                    match d:
                        case Dir(-1, -1) | Dir(1, 1):
                            s[ni][nj] = "x" if s[ni][nj] == "/" else "\\"
                        case Dir(-1, 0) | Dir(1, 0):
                            s[ni][nj] = "+" if s[ni][nj] == "-" else "|"
                        case Dir(-1, 1) | Dir(1, -1):
                            s[ni][nj] = "x" if s[ni][nj] == "\\" else "/"
                        case Dir(0, -1) | Dir(0, 1):
                            s[ni][nj] = "+" if s[ni][nj] == "|" else "-"

        return s

    def dump(self) -> None:
        RED = "\033[91m"
        BLUE = "\033[94m"
        RESET = "\033[0m"
        colored = self.to_chars()
        for i in range(len(colored)):
            for j in range(len(colored[i])):
                match colored[i][j]:
                    case "@":
                        colored[i][j] = RED + "@" + RESET
                    case "#":
                        colored[i][j] = BLUE + "#" + RESET
        print(" ", " ".join(str(i) for i in range(self.width)))
        for i, row in enumerate(colored):
            print(str(i // 2) if i % 2 == 0 else " ", "".join(row))

    def score(self, color: Color) -> int:
        history = self.red_history if color == Color.Red else self.blue_history
        graph = Graph(self.width)
        for move in history:
            graph.apply_move(move)
        return graph.score()

    def playout(self, init_turn: Color, rng: np.random.Generator) -> Optional[Color]:
        next_turn = opponent_of(init_turn)
        while not self.is_gameover():
            for turn in [init_turn, next_turn]:
                possible_moves = self.possible_moves(turn)
                if len(possible_moves) > 0:
                    self.apply_move(rng.choice(possible_moves), turn)
        red_score = self.score(Color.Red)
        blue_score = self.score(Color.Blue)
        return (
            Color.Red
            if red_score > blue_score
            else Color.Blue
            if blue_score > red_score
            else None
        )
