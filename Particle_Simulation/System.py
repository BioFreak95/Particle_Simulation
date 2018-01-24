import numpy as np
from Particle_Simulation.Energy import Energy
from Particle_Simulation.Neighbourlist import Neighbourlist


class System:
    cell_neighbour_list = None

    # Test Positions are same dimension

    def __init__(self, particles, parameters):
        self.particles = particles
        self.energy = Energy()
        self.k_vectors = [1, 1, 1]

        x = []
        for i in range(len(particles)):
            x.append(particles[i].position)
        y = np.array(x)

        self.neighbourlist = Neighbourlist(particles=y, Box=parameters.box, rc=parameters.cutoff_radius)

        if System.cell_neighbour_list is None:
            System.cell_neighbour_list = self.neighbourlist.calc_cell_neighbours()