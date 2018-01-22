import numpy as np
import math
from numba import jitclass
from numba import float64, float32, int8, int32, int16, int64

from System import System
from Energy import Energy


specs = [
    ('particle_positions', float64[:, :]),
    ('cell_list', int32[:]),
    ('particle_neighbour_list', int32[:]),

    ('cell_neighbour_list', int32[:, :, :]),

    ('box', float64[:]),
    ('cutoff_radius', float32),
    ('es_sigma', float32),

    ('charges', float32[:]),
    ('lj_sigmas', float32[:]),
    ('lj_epsilons', float32[:]),

    ('VACUUM_PERMITTIVITY', float32),

]

cell_shift_list = np.array([
    [0, 1, -1, 1, -1, 1, -1, 0, 0, 0, 1, -1, 1, -1, 1, -1, 0, 0, 0, 1, -1, 1, -1, 1, -1, 0, 0],
    [0, 0, 0, 1, 1, -1, -1, 1, -1, 0, 0, 0, 1, 1, -1, -1, 1, -1, 0, 0, 0, 1, 1, -1, -1, 1, -1],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, -1, -1, -1, -1, -1, -1, -1, -1, -1]])


@jitclass(specs)
class EnergyCalculator:

    # - - - constructor - - - #

    def __init__(self, box, cutoff_radius, es_sigma, charges, lj_sigmas, lj_epsilons):

        # variables (change every iteration)
        self.particle_positions = np.zeros((1, 1), dtype=np.float64)
        self.cell_list = np.zeros(1, dtype=np.int32)
        self.particle_neighbour_list = np.zeros(1, dtype=np.int32)

        # simulation specific constants (defined once when instantiating this class)
        self.box = box.astype(np.float64)
        self.cutoff_radius = cutoff_radius
        self.es_sigma = es_sigma
        self.charges = charges
        self.lj_sigmas = lj_sigmas
        self.lj_epsilons = lj_epsilons

        self.cell_neighbour_list = np.zeros((1, 1, 1), dtype=np.int32)

        # constants (same every simulation)
        self.VACUUM_PERMITTIVITY = 1

    # - - - public methods - - - #

    def set_system(self, particle_positions, cell_list, particle_neighbour_list):
        self.particle_positions = particle_positions
        self.cell_list = cell_list
        self.particle_neighbour_list = particle_neighbour_list

    def calculate_shortranged_energy_2(self):

        lj_energy = 0
        short_ranged_energy = 0

        for i in range(0, len(self.particle_positions) - 1):
            for j in range(i + 1, len(self.particle_positions)):
                particle_distance = self._calculate_norm(self._wrap_distance(self.particle_positions[i] - self.particle_positions[j]))

                if particle_distance < self.cutoff_radius:
                    lj_energy += self._calculate_lj_potential(i, j, particle_distance)
                    short_ranged_energy += self._calculate_shortranged_potential(i, j, particle_distance)

        short_ranged_energy *= 1 / (8 * np.pi * self.VACUUM_PERMITTIVITY)
        return [lj_energy, short_ranged_energy]

    def calculate_selfinteraction_energy(self):

        summation = 0
        prefactor = 1 / (2 * self.VACUUM_PERMITTIVITY * self.es_sigma * (2 * np.pi) ** (3 / 2))

        for i in range(0, len(self.particle_positions)):
            summation += self._calculate_selfinteraction_potential(i)
        selfinteraction_energy = prefactor * summation

        return selfinteraction_energy

    # - - - private methods - - - #

    def _calculate_lj_potential(self, particle_index_1, particle_index_2, particle_distance):

        sigma = self._determine_sigma(particle_index_1, particle_index_2)
        epsilon = self._determine_epsilon(particle_index_1, particle_index_2)

        attractive_term = (sigma / particle_distance) ** 6
        repulsive_term = attractive_term ** 2
        lj_potential = 4 * epsilon * (repulsive_term - attractive_term)

        return lj_potential

    def _calculate_shortranged_potential(self, particle_index_1, particle_index_2, particle_distance):

        charge_1 = self.charges[particle_index_1]
        charge_2 = self.charges[particle_index_2]

        short_ranged_potential = ((charge_1 * charge_2) / (particle_distance)) * math.erfc((particle_distance) / (np.sqrt(2) * self.es_sigma))

        return short_ranged_potential

    def _calculate_selfinteraction_potential(self, particle_index):
        return self.charges[particle_index] ** 2

    def _wrap_distance(self,distance):

        for i in range(len(distance)):
            while distance[i] >= 0.5 * self.box[i]:
                distance[i] -= self.box[i]
            while distance[i] < -0.5 * self.box[i]:
                distance[i] += self.box[i]

        return distance

    def _calculate_norm(self, distance):

        summation = 0
        for i in range(len(distance)):
            summation += distance[i] ** 2
        distance = np.sqrt(summation)
        return distance

    def _determine_sigma(self, particle_index_1, particle_index_2):

        sigma = 0.5 * (self.lj_sigmas[particle_index_1] + self.lj_sigmas[particle_index_2])
        return sigma

    def _determine_epsilon(self, particle_index_1, particle_index_2):

        epsilon = np.sqrt(self.lj_epsilons[particle_index_1] * self.lj_epsilons[particle_index_2])
        return epsilon

    def calculate_shortranged_energy(self):

        lj_energy = 0
        short_ranged_energy = 0
        neighbour_cell_number = 3 ** len(self.particle_positions[0])

        for i in range(len(self.cell_list)):
            particle_index_1 = self.cell_list[i]

            while particle_index_1 != -1:

                for k in range(neighbour_cell_number):
                    cell_index = self.cell_neighbour_list[k][i][0]
                    particle_index_2 = self.cell_list[cell_index]

                    while particle_index_2 != -1:

                        if self.cell_neighbour_list[k][i][1] == 0:
                            if particle_index_1 < particle_index_2:

                                particle_distance = self._calculate_norm(self.particle_positions[particle_index_1] - self.particle_positions[particle_index_2])
                                if particle_distance < self.cutoff_radius:
                                    lj_energy += self._calculate_lj_potential(particle_index_1, particle_index_2, particle_distance)
                                    short_ranged_energy += self._calculate_shortranged_potential(particle_index_1, particle_index_2, particle_distance)

                        elif self.cell_neighbour_list[k][i][1] != 0:
                            if particle_index_1 < particle_index_2:

                                box_shift = self._determine_box_shift(i, k)
                                particle_distance = self._calculate_norm(self.particle_positions[particle_index_1] - (self.particle_positions[particle_index_2] + box_shift))

                                if particle_distance < self.cutoff_radius:

                                    lj_energy += self._calculate_lj_potential(particle_index_1, particle_index_2, particle_distance)
                                    short_ranged_energy += self._calculate_shortranged_potential(particle_index_1, particle_index_2, particle_distance)

                        particle_index_2 = self.particle_neighbour_list[particle_index_2]
                particle_index_1 = self.particle_neighbour_list[particle_index_1]

        short_ranged_energy *= 1 / (8 * np.pi * self.VACUUM_PERMITTIVITY)
        return [lj_energy, short_ranged_energy]

    def _determine_box_shift(self, cell_index, cell_neighbour_index):

        box_shift = np.zeros((len(self.box)))
        if self.cell_neighbour_list[cell_neighbour_index][cell_index][1] != 0:
            for i in range(len(self.box)):
                if cell_shift_list[i][cell_neighbour_index] == 1:
                    box_shift[i] = self.box[i]
                elif cell_shift_list[i][cell_neighbour_index] == -1:
                    box_shift[i] = -self.box[i]
                else:
                    continue

        return box_shift



















'''
    def calculate_overall_energy(self, system):

        overall_energy = Energy()

        short_ranged_energy = self.calculate_shortranged_energy(system)
        overall_energy.lj_energy = short_ranged_energy[0]
        overall_energy.es_shortranged_energy = short_ranged_energy[1]
        overall_energy.es_selfinteraction_energy = self.calculate_shortranged_energy(system)

        return overall_energy
'''
