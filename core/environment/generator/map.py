from numpy import array
from random import randint
from typing import Tuple


def generate_map(matrix: array) -> Tuple[array, int, int]:
    probabilities: list = [probability for line in matrix for probability in line]
    probabilities_times_random_factor: list = [
        (randint(1, 100) * probability) / 100 for probability in probabilities
    ]

    person_position: int = probabilities_times_random_factor.index(
        max(probabilities_times_random_factor)
    )

    position_matrix: list = [[0 for _ in range(0, matrix.shape[1])] for _ in matrix]

    line: int = person_position // matrix.shape[1]
    column: int = person_position % matrix.shape[1]

    position_matrix[line][column] = "P"

    position_matrix[0][0] = "X" if position_matrix[0][0] != "P" else "PX"

    return array(position_matrix), column, line
