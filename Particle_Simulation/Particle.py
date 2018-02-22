import numpy as np

'''TODO : Check if position are floats64'''


class Particle:
    def __init__(self, position):
        if not isinstance(position, np.ndarray):
            raise TypeError('position have to be an array')
        self.position = position
