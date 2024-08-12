import separo
import time
import copy
from typing import Optional, Self
from dataclasses import dataclass, field
import numpy as np


@dataclass
class RandomPlayer:
    color: separo.Color
    rng: np.random.Generator

    def next_move(self, board: separo.Board) -> Optional[separo.Move]:
        possible_moves = board.possible_moves(self.color)
        if len(possible_moves) == 0:
            return None
        return self.rng.choice(possible_moves)


@dataclass
class HumanPlayer:
    color: separo.Color

    def next_move(self, board: separo.Board) -> Optional[separo.Move]:
        possible_moves = board.possible_moves(self.color)
        if len(possible_moves) == 0:
            print("You cannot move. Passed.")
            return None
        board.dump()
        print("Choose your next move by the index...")
        for i, move in enumerate(possible_moves):
            print(f"{i}: {move}")
        while True:
            try:
                idx = int(input())
                assert 0 <= idx < len(possible_moves)
                return possible_moves[idx]
            except ValueError:
                print("Invalid index. Try again.")
                continue


@dataclass
class MCNode:
    next_move: separo.Move
    next_board: separo.Board
    wins: int


@dataclass
class NaiveMCPlayer:
    color: separo.Color
    time_limit: float
    rng: np.random.Generator

    def next_move(self, board: separo.Board) -> Optional[separo.Move]:
        candidates: list[MCNode] = []
        for move in board.possible_moves(self.color):
            cand_board = copy.deepcopy(board)
            cand_board.apply_move(move, self.color)
            candidates.append(MCNode(move, cand_board, 0))
        if len(candidates) == 0:
            return None
        stop = time.time() + self.time_limit
        samples: int = 0
        while time.time() < stop:
            for candidate in candidates:
                tmp = copy.deepcopy(candidate.next_board)
                if tmp.playout(self.color, self.rng) == self.color:
                    candidate.wins += 1
            samples += 1
        opt_cand = max(candidates, key=lambda candidate: candidate.wins)
        # print(f"NaiveMC({self.color}): sampled {samples * len(candidates)}, estimated win rate = {opt_cand.wins/samples}")
        return opt_cand.next_move


@dataclass
class UCTNode:
    color: separo.Color
    board: separo.Board
    last_move: Optional[separo.Move]
    wins: int = 0
    loses: int = 0
    samples: int = 0
    children: list[Self] = field(default_factory=list)

    def win_rate(self) -> float:
        return 0.5 if self.samples == 0 else self.wins / self.samples

    def lose_rate(self) -> float:
        return 0.5 if self.samples == 0 else self.loses / self.samples

    def ucb1(self, coef: float, logn: float) -> float:
        return (
            float("inf")
            if self.samples == 0
            else self.win_rate() + coef * np.sqrt(logn / self.samples)
        )

    def expand_node(self) -> None:
        assert len(self.children) == 0
        for possible_move in self.board.possible_moves(self.color):
            possible_board = copy.deepcopy(self.board)
            possible_board.apply_move(possible_move, self.color)
            child = UCTNode(
                separo.opponent_of(self.color), possible_board, possible_move
            )
            self.children.append(child)
        if len(self.children) == 0:
            child = UCTNode(
                separo.opponent_of(self.color), copy.deepcopy(self.board), None
            )
            self.children.append(child)


@dataclass
class PUCTMCPlayer:
    color: separo.Color
    time_limit: float
    rng: np.random.Generator
    ucb1_coeff: float
    expand_threshold: int
    root: UCTNode

    def __init__(
        self,
        color: separo.Color,
        time_limit: float,
        rng: np.random.Generator,
        ucb1_coeff: float,
        expand_threshold: int,
        width: int,
    ) -> None:
        self.color = color
        self.time_limit = time_limit
        self.rng = rng
        self.ucb1_coeff = ucb1_coeff
        self.expand_threshold = expand_threshold

        match color:
            case separo.Color.Red:
                ancester = UCTNode(separo.Color.Blue, separo.Board(width), None)
                root = UCTNode(separo.Color.Red, separo.Board(width), None)
                ancester.children.append(root)
                self.root = ancester
            case separo.Color.Blue:
                self.root = UCTNode(separo.Color.Red, separo.Board(width), None)
                self.root.expand_node()

    def next_move(self, board: separo.Board) -> Optional[separo.Move]:
        if not board.can_move(self.color):
            return None
        self.root = list(
            filter(lambda child: child.board == board, self.root.children)
        )[0]
        stop = time.time() + self.time_limit
        while time.time() < stop:
            node = self.root
            logn = np.log(node.samples + 1)  # avoid log(0)
            history: list[UCTNode] = [node]
            while len(node.children) > 0:
                node = max(
                    node.children, key=lambda child: child.ucb1(self.ucb1_coeff, logn)
                )
                history.append(node)

            wins = copy.deepcopy(node.board).playout(node.color, self.rng)
            for ancestor in history:
                ancestor.samples += 1
                if wins == ancestor.color:
                    ancestor.loses += 1
                if wins == separo.opponent_of(ancestor.color):
                    ancestor.wins += 1

            if self.expand_threshold <= node.samples:
                if not node.board.is_gameover():
                    node.expand_node()

        print(
            list(
                map(
                    lambda node: (
                        node.samples,
                        node.ucb1(self.ucb1_coeff, np.log(self.root.samples)),
                    ),
                    self.root.children,
                )
            )
        )

        self.root = max(self.root.children, key=lambda node: node.samples)
        # print(
        #     f"{self.color}, estimated (win, lose) rate = ({self.root.win_rate()}, {self.root.lose_rate()})"
        # )

        if len(self.root.children) == 0 and not self.root.board.is_gameover():
            self.root.expand_node()
            print(
                f"self.root.samples = {self.root.samples}: root.children is empty. Too short time limit?"
            )

        return self.root.last_move


class AlphaZeroPlayer:
    pass
