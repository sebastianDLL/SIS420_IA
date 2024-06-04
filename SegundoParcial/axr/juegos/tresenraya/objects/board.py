import numpy as np

class Board():
    def __init__(self):
        self.state = np.zeros((4,4))

    def valid_moves(self):
        return [(i, j) for j in range(4) for i in range(4) if self.state[i, j] == 0]

    def update(self, symbol, row, col):
        if self.state[row, col] == 0:
            self.state[row, col] = symbol
        else:
            raise ValueError("¡Movimiento ilegal!")

    def is_game_over(self):
        # Comprobar filas y columnas
        if (self.state.sum(axis=0) == 4).sum() >= 1 or (self.state.sum(axis=1) == 4).sum() >= 1:
            return 1
        if (self.state.sum(axis=0) == -4).sum() >= 1 or (self.state.sum(axis=1) == -4).sum() >= 1:
            return -1 
        # Comprobar diagonales
        diag_sums = [
            sum([self.state[i, i] for i in range(4)]),
            sum([self.state[i, 4 - i - 1] for i in range(4)]),
        ]
        if 4 in diag_sums:
            return 1
        if -4 in diag_sums:
            return -1        
        # Empate
        if len(self.valid_moves()) == 0:
            return 0
        # Seguir jugando
        return None

    def reset(self):
        self.state = np.zeros((4, 4))