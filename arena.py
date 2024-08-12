from dataclasses import dataclass
import separo
from typing import Optional
import ray
from ray.experimental import tqdm_ray

remote_tqdm = ray.remote(tqdm_ray.tqdm)


@ray.remote
def play(
    # bar: tqdm_ray.tqdm,
    width: int,
    red,
    blue,
    verbose=False,
) -> Optional[separo.Color]:
    board = separo.Board(width)

    while not board.is_gameover():
        red_action = red.next_move(board)
        if red_action is not None:
            board.apply_move(red_action, separo.Color.Red)
        if verbose:
            print(f"Red: {red_action}")
            board.dump()
        blue_action = blue.next_move(board)
        if blue_action is not None:
            board.apply_move(blue_action, separo.Color.Blue)
        if verbose:
            print(f"Blue: {blue_action}")
            board.dump()

    red_score = board.score(separo.Color.Red)
    blue_score = board.score(separo.Color.Blue)
    winner = (
        separo.Color.Red
        if red_score > blue_score
        else separo.Color.Blue
        if blue_score > red_score
        else None
    )
    if verbose:
        print(
            f"Winner: {winner}; Red({red_score}) vs Blue({blue_score})",
        )
    # bar.update.remote(1)
    return winner


class Arena:
    def __init__(self, width: int, red, blue):
        self.width = width
        self.red = red
        self.blue = blue

    @dataclass
    class Results:
        red_wins: int = 0
        blue_wins: int = 0
        draws: int = 0

    def play_matchs(self, n_matches: int = 100, verbose=False) -> Results:
        results = self.Results()
        # bar = remote_tqdm.remote(total=n_matches)
        winners = [
            play.remote(
                # bar,
                self.width,
                self.red,
                self.blue,
                verbose=verbose,
            )
            for _ in range(n_matches)
        ]
        for winner in ray.get(winners):
            if winner == separo.Color.Red:
                results.red_wins += 1
            elif winner == separo.Color.Blue:
                results.blue_wins += 1
            else:
                results.draws += 1
        # bar.close()
        return results
