import numpy as np
from IPython.display import clear_output
import time
import seaborn as sns
import matplotlib.pyplot as plt


def update_board(current_board):
    """
    Execute one iteration of Conway's Game of Life on a binary grid.

    Parameters
    ----------
    current_board : numpy.ndarray
        A 2D binary NumPy array representing the current state of the game board.
        Each element should be 1 for a live cell and 0 for a dead cell.

    Returns
    -------
    updated_board : numpy.ndarray
        A 2D binary NumPy array of the same shape as `current_board`, representing
        the next state of the board after applying the Game of Life rules.


    """
    # Pad the board to handle edge cells
    padded_board = np.pad(current_board, pad_width=1, mode='constant', constant_values=0)
    updated_board = np.zeros_like(current_board)

    # Iterate through each cell
    for i in range(1, padded_board.shape[0] - 1):
        for j in range(1, padded_board.shape[1] - 1):
            # Count live neighbors
            neighborhood = padded_board[i-1:i+2, j-1:j+2]
            live_neighbors = np.sum(neighborhood) - padded_board[i, j]

            # Apply rules
            if padded_board[i, j] == 1:
                # Cell is alive
                if live_neighbors < 2 or live_neighbors > 3:
                    updated_board[i-1, j-1] = 0  # Dies
                else:
                    updated_board[i-1, j-1] = 1  # Lives
            else:
                # Cell is dead
                if live_neighbors == 3:
                    updated_board[i-1, j-1] = 1  # Becomes alive

    return updated_board


def show_game(game_board, n_steps=10, pause=0.5):
    """
    Show `n_steps` of Conway's Game of Life, given the `update_board` function.

    Parameters
    ----------
    game_board : numpy.ndarray
        A binary array representing the initial starting conditions for Conway's Game of Life. In this array, ` represents a "living" cell and 0 represents a "dead" cell.
    n_steps : int, optional
        Number of game steps to run through, by default 10
    pause : float, optional
        Number of seconds to wait between steps, by default 0.5
    """
    for step in range(n_steps):
        clear_output(wait=True)

        # update board
        game_board = update_board(game_board)

        # show board
        sns.heatmap(game_board, cmap='plasma', cbar=False, square=True)
        plt.title(f'Board State at Step {step + 1}')
        plt.show()

        # wait for the next step
        if step + 1 < n_steps:
            time.sleep(pause)
