import numpy as np

'''TODO : Check if position are floats64'''
class Particle:
    def __init__(self, type_index, position):
        if not isinstance(type_index, int):
            raise TypeError('type_index have to be a integer')
        if not isinstance(position, np.ndarray):
            raise TypeError('position have to be an array')
        self.type_index = type_index
        self.position = position