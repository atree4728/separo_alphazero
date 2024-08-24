from typing import Self, Optional
import separo
import copy
import torch
import torch.nn as nn
import torch.optim as optim
from dataclasses import dataclass, field
import collections
import numpy as np


class ResBlock(nn.Module):
    def __init__(self, filters: int) -> None:
        super(Self, self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(
                in_channels=filters, out_channels=filters, kernel_size=3, padding="same"
            ),
            nn.BatchNorm2d(filters),
            nn.ReLU,
            nn.Conv2d(
                in_channels=filters, out_channels=filters, kernel_size=3, padding="same"
            ),
            nn.BatchNorm2d(filters),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return nn.functional.relu(self.model(x) + x)


class NNet(nn.Module):
    def __init__(self, width: int, filters: int = 256) -> None:
        super(Self, self).__init__()
        self.body = nn.Sequential(
            nn.Conv2d(
                in_channels=2, out_channels=filters, kernel_size=3, padding="same"
            ),
            nn.BatchNorm2d(num_features=filters),
            nn.ReLU,
            # 10 ResBlock
            ResBlock(filters),
            ResBlock(filters),
            ResBlock(filters),
            ResBlock(filters),
            ResBlock(filters),
            ResBlock(filters),
            ResBlock(filters),
            ResBlock(filters),
            ResBlock(filters),
            ResBlock(filters),
        )
        self.policy_head = nn.Sequential(
            nn.Conv2d(
                in_channels=filters, out_channels=2, kernel_size=3, padding="same"
            ),
            nn.BatchNorm2d(num_features=2),
            nn.ReLU,
            nn.Flatten,
            nn.Linear((3 * self.width) ** 2, self.width**2 * 8),
            nn.Softmax,
        )
        self.value_head = nn.Sequential(
            nn.Conv2d(
                in_channels=filters, out_channels=1, kernel_size=3, padding="same"
            ),
            nn.BatchNorm2d(num_features=1),
            nn.ReLU,
            nn.Flatten,
            nn.Linear((3 * self.width) ** 2, filters),
            nn.ReLU,
            nn.Linear(filters, 1),
            nn.Tanh,
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if len(x.shape) == 3:
            x = torch.unsqueeze(x, 0)
        # tensor: batch_size * 2 * 3w * 3w
        # tuple(1, width * width * 8); 8 = Move.size()
        x = self.body(x)
        return (self.policy_head(x), self.value_head(x))


class Config:
    def __init__(self, width: int, rng: np.random.Generator):
        ### Self-Play
        self.width = width
        self.num_actors = 5000
        self.action_space = width * width * 8

        self.num_sampling_moves = 10
        self.num_simulations = 800

        # Root prior exploration noise.
        self.root_dirichlet_alpha = 0.03
        self.root_exploration_fraction = 0.25

        # UCB formula
        self.c_puct = 1.0

        ### Training
        self.training_steps = int(700e3)
        self.checkpoint_interval = int(1e3)
        self.buffer_size = int(1e6)
        self.batch_size = 4096

        # Schedule for chess and shogi, Go starts at 2e-2 immediately.
        self.learning_rate_schedule = {0: 2e-2, 100e3: 2e-2, 300e3: 2e-3, 500e3: 2e-4}

        self.rng = rng


@dataclass
class Node:
    color: separo.Color
    board: separo.Board
    prior: float
    value: float
    visit_count: int = 0
    value_sum: float = 0
    children: Optional[dict[int, Self]] = None

    def expand_node(self, nnet: NNet) -> None:
        assert self.children is None
        self.children = {}
        state = self.board.encode(self.color)
        priors, value = nnet.forward(state)
        self.value = value.item()
        possible_moves = self.board.possible_moves(self.color)
        if len(possible_moves) == 0:
            child = Node(separo.opponent_of(self.color), copy.deepcopy(self.board), 1)
            self.children[None] = child
            return
        for move in self.board.possible_moves(self.color):
            id = separo.into_id(move)
            board = copy.deepcopy(self.board)
            board.apply_move(move, self.color)
            child = Node(separo.opponent_of(self.color), board, priors[id])
            self.children[id] = child


@dataclass
class PVMonteCarlo:
    nnet: NNet
    config: Config

    def search(self, root: Node) -> np.array:
        if len(root.children) == 0:
            root.expand_node(self.nnet)

        possible_moves = root.board.possible_moves(root.color)
        possible_move_ids = [
            separo.into_id(move, self.config.width) for move in possible_moves
        ]
        noises = self.config.rng.dirichlet(
            [self.config.root_dirichlet_alpha] * len(possible_moves)
        )
        for id, noise in zip(possible_move_ids, noises):
            root.children[id].prior = (
                1 - self.config.root_exploration_fraction
            ) * root.children[id].prior + self.config.root_exploration_fraction * noise

        for _ in range(self.config.num_simulations):
            node = copy.deepcopy(root)
            history: list[tuple[Node, int]] = []
            while node.children is not None:
                U = np.array(
                    [
                        self.config.c_puct
                        * node.children[id].prior
                        * np.sqrt(node.visit_count)
                        / (1 + node.children[id].visit_count)
                        for id in possible_move_ids
                    ]
                )
                Q = np.array(
                    [
                        node.children[id].value_sum / node.children[id].visit_count
                        if node.children[id].visit_count > 0
                        else 0
                        for id in possible_move_ids
                    ]
                )
                scores = U + Q
                next_move_id_ind = self.config.rng.choice(
                    np.where(scores == scores.max())[0]
                )
                next_move_id = possible_moves[next_move_id_ind]
                history.append((node, next_move_id))
                node = node.children[next_move_id]

            if node.board.is_gameover():
                winner = node.board.winner()
                value = (
                    1
                    if winner == node.color
                    else -1
                    if winner == separo.opponent_of(node.color)
                    else 0
                )
            else:
                node.expand_node(self.nnet)
                value = node.value

            for node, move_id in history:
                node.children[move_id].visit_count += 1
                node.children[move_id].visit_count += value * (
                    1 if node.color == root.color else -1
                )

        mcts_policy = np.array(
            [
                root.children[move_id].visit_count if move_id in root.children else 0
                for move_id in range(self.config.action_space)
            ]
        )
        mcts_policy /= sum(mcts_policy)
        return mcts_policy


@dataclass
class Sample:
    state: torch.Tensor  # including color info
    policy: torch.Tensor
    reward: int


@dataclass
class ReplayBuffer:
    rng: np.random
    buffer_size: int
    buffer: collections.deque[Sample] = field(default_factory=collections.deque)

    def __post_init__(self):
        self.buffer = collections.deque(maxlen=self.buffer_size)

    def sample_batch(self):
        indices = self.rng.choice(len(self.buffer), size=self.buffer_size)
        samples = torch.Tensor([self.buffer[idx] for idx in indices])
        return tuple(samples.transpose())

    def save(
        self,
        board: separo.Board,
        color: separo.Color,
        policy: torch.Tensor,
        reward: int,
    ):
        state = board.encode(color)
        self.buffer.append(Sample(state, policy, reward))


def selfplay(nnet: NNet, config: Config):
    turn = separo.Color.Red
    board = separo.Board(config.width)

    mcts = PVMonteCarlo(nnet, config)
    samples: list[Sample] = []

    it = 0
    while not board.is_gameover():
        mcts_policy = mcts.search(board, turn, config)
        next_move = (
            config.rng.choice(config.action_space, p=mcts_policy)
            if it <= config.num_sampling_moves
            else config.rng.choice(np.where(mcts_policy == max(mcts_policy))[0])
        )
        board.apply_move(next_move, turn)
        turn = separo.opponent_of(turn)
        samples.append(
            Sample(board.encode(turn), mcts_policy, turn.value)
        )  # turn.value: rewrite to reward later
        it += 1

    winner = board.winner()
    reward = (
        1 if winner == separo.Color.Red else -1 if winner == separo.Color.Blue else 0
    )
    for sample in samples:
        sample.reward = reward * (1 if sample.turn == separo.Color.Red else -1)

    return samples


def train(config: Config):
    pass
