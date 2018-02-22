import numpy as np
import math
from numba import jitclass
from numba import float64, float32, int32, int64

specs = [
    ('particle_positions', float64[:, :]),
    ('cell_list', int64[:]),
    ('particle_neighbour_list', int64[:]),

    ('cell_neighbour_list', int32[:, :, :]),

    ('box', float64[:]),
    ('cutoff_radius', float32),
    ('es_sigma', float32),
    ('k_vector', int64[:, :]),

    ('charges', float32[:]),
    ('lj_sigmas', float64[:]),
    ('lj_epsilons', float64[:]),

    ('UNIT_PREFACTOR', float64),
]

cell_shift_list = np.array([
    [0, 1, -1, 1, -1, 1, -1, 0, 0, 0, 1, -1, 1, -1, 1, -1, 0, 0, 0, 1, -1, 1, -1, 1, -1, 0, 0],
    [0, 0, 0, 1, 1, -1, -1, 1, -1, 0, 0, 0, 1, 1, -1, -1, 1, -1, 0, 0, 0, 1, 1, -1, -1, 1, -1],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, -1, -1, -1, -1, -1, -1, -1, -1, -1]])


@jitclass(specs)
class EnergyCalculator:

    # - - - constructor - - - #

    def __init__(self, box, cutoff_radius, es_sigma, charges, lj_sigmas, lj_epsilons, k_vector):

        # system specific variables (change every iteration)
        self.particle_positions = np.zeros((1, 1), dtype=np.float64)
        self.cell_list = np.zeros(1, dtype=np.int64)
        self.particle_neighbour_list = np.zeros(1, dtype=np.int64)

        # simulation specific constants (defined once when instantiating this class)
        self.box = box.astype(np.float64)
        self.cutoff_radius = cutoff_radius
        self.es_sigma = es_sigma
        self.charges = charges.astype(np.float32)
        self.lj_sigmas = lj_sigmas.astype(np.float64)
        self.lj_epsilons = lj_epsilons.astype(np.float64)
        self.k_vector = k_vector

        self.cell_neighbour_list = np.zeros((1, 1, 1), dtype=np.int32)

        # constants (same every simulation)
        VACUUM_PERMITTIVITY =  8.854187817 * 10 ** (-12)
        ELEMENTARY_CHARGE_SQUARED = 1.602176620898 * 10 ** (-38)

        self.UNIT_PREFACTOR = 10 ** (9) * ELEMENTARY_CHARGE_SQUARED / VACUUM_PERMITTIVITY # same for selfinteraction

    # - - - public methods - - - #

    def set_system(self, particle_positions, cell_list, particle_neighbour_list):
        self.particle_positions = particle_positions
        self.cell_list = cell_list
        self.particle_neighbour_list = particle_neighbour_list

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
                            if particle_index_1 != particle_index_2:

                                box_shift = self._determine_box_shift(i, k)
                                particle_distance = self._calculate_norm(self.particle_positions[particle_index_1] - (self.particle_positions[particle_index_2] + box_shift))

                                if particle_distance != self.cutoff_radius:

                                    lj_energy += self._calculate_lj_potential(particle_index_1, particle_index_2, particle_distance)
                                    short_ranged_energy += self._calculate_shortranged_potential(particle_index_1, particle_index_2, particle_distance)

                        particle_index_2 = self.particle_neighbour_list[particle_index_2]
                particle_index_1 = self.particle_neighbour_list[particle_index_1]

        short_ranged_energy *= self.UNIT_PREFACTOR / (8 * np.pi)
        return [lj_energy, short_ranged_energy]

    def calculate_longranged_energy(self):

        particle_number = len(self.particle_positions)
        long_ranged_energy = 1

        for i in range(len(self.k_vector)):
            s_k = 0
            k = self.k_vector[i]
            kn = self._calculate_norm(k)

            for j in range(particle_number):
                sin_contribution = self.charges[j] * np.sin(self._calculate_dot_product(k, self.particle_positions[j]))
                cos_contribution = self.charges[j] * np.cos(self._calculate_dot_product(k, self.particle_positions[j]))
                s_k += sin_contribution ** 2 + cos_contribution ** 2

            long_ranged_energy += s_k * (np.e ** (-((self.es_sigma ** 2) * (kn ** 2)) / 2)) / (kn ** 2)
        long_ranged_energy *= self.UNIT_PREFACTOR / np.prod(self.box)

        return long_ranged_energy

    def calculate_selfinteraction_energy(self):

        summation = 0
        prefactor = self.UNIT_PREFACTOR / (2 * self.es_sigma * (2 * np.pi) ** (3 / 2))

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

        short_ranged_potential = ((charge_1 * charge_2) / particle_distance) * math.erfc((particle_distance) / (np.sqrt(2) * self.es_sigma))

        return short_ranged_potential

    def _calculate_selfinteraction_potential(self, particle_index):
        return self.charges[particle_index] ** 2

    def _calculate_dot_product(self, vector_1, vector_2):

        if len(vector_1) != len(vector_2):
            return 0

        dot_product = 0
        for i in range(len(vector_1)):
            dot_product += vector_1[i] * vector_2[i]

        return dot_product

    def _wrap_distance(self,distance):

        for i in range(len(distance)):
            while distance[i] >= 0.5 * self.box[i]:
                distance[i] -= self.box[i]
            while distance[i] < -0.5 * self.box[i]:
                distance[i] += self.box[i]

        return distance

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

    def _calculate_norm(self, vector):

        summation = 0
        for i in range(len(vector)):
            summation += vector[i] ** 2
        distance = np.sqrt(summation)
        return distance

    def _determine_sigma(self, particle_index_1, particle_index_2):

        sigma = 0.5 * (self.lj_sigmas[particle_index_1] + self.lj_sigmas[particle_index_2])
        return sigma

    def _determine_epsilon(self, particle_index_1, particle_index_2):

        epsilon = np.sqrt(self.lj_epsilons[particle_index_1] * self.lj_epsilons[particle_index_2])
        return epsilon

    def calculate_shortranged_energy_naive(self):

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