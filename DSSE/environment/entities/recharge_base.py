import random

class RechargeBase:
    def __init__(self, grid_size, position=None):
        self.grid_size = grid_size

        if position is None:
            self.position = (0, 0)
        else:
            self.position = position

    # Get the current position of the recharge base
    def get_position(self):
        return self.position
