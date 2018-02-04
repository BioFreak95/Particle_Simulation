import numpy as np
from Energy import Energy
from Neighbourlist import Neighbourlist


class System:
    cell_neighbour_list = None

    # Test Positions are same dimension

    def __init__(self, particles, parameters):
        self.particles = particles
        self.energy = Energy()
        self.k_vectors = [1, 1, 1]

        particle_positions = self.get_particle_position_array()

        self.neighbourlist = Neighbourlist(particles=particle_positions, Box=parameters.box, rc=parameters.cutoff_radius)

        if System.cell_neighbour_list is None:
            System.cell_neighbour_list = self.neighbourlist.calc_cell_neighbours()

    def get_particle_position_array(self):

        x = []
        for i in range(len(self.particles)):
            x.append(self.particles[i].position)
        y = np.array(x)

        return y
