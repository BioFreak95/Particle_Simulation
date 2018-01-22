import numpy as np
from numba import jitclass
from numba import float64, float32, int8, int32, int16, int64

specs = [
    ('particle_positions', float64[:, :]),
    ('box_space', float32[:]),
    ('cutoff', float32),
    ('particle_number', int32),
    ('dim', int8),
    ('cell_number', int16[:]),
    ('cell_space', float32[:]),
    ('total_cell_number', int32),
    ('cell_list', int32[:]),
    ('particle_neighbour_list', int32[:]),
]


@jitclass(specs)
class Neighbourlist:
    def __init__(self, particles, Box, rc):

        self.particle_positions = particles.astype(np.float64)
        self.box_space = Box.astype(np.float32)
        self.cutoff = rc

        self.particle_number = len(particles)

        self.dim = len(self.particle_positions[0])
        self.cell_number = np.zeros(self.dim, dtype=np.int16)
        self.cell_space = np.zeros(self.dim, dtype=np.float32)

        for i in range(len(self.box_space)):
            self.cell_number[i] = np.floor(self.box_space[i] / self.cutoff)
            self.cell_space[i] = self.box_space[i] / self.cell_number[i]

        self.total_cell_number = np.prod(self.cell_number)
        self.cell_list = np.zeros(self.total_cell_number, dtype=np.int32) - 1
        self.particle_neighbour_list = np.zeros(self.particle_number, dtype=np.int32) - 1

        self.construct_neighbourlist()

    def construct_neighbourlist(self):

        # wipe neighborlists
        self.cell_list = np.zeros(self.total_cell_number, dtype=np.int32) - 1
        self.particle_neighbour_list = np.zeros(self.particle_number, dtype=np.int32) - 1
        # create interim position array to operate on (so that particle_positions stays untouched)
        shifted_particle_positions = self.particle_positions

        for i in range(self.particle_number):

            particle_cell_location = []
            for a in range(len(self.particle_positions[i])):
                shifted_particle_positions[i][a] = self.periodic_box_shift(self.particle_positions[i][a],
                                                                        self.box_space[a])
                particle_cell_location.append(np.floor(shifted_particle_positions[i][a] / self.cell_space[a]))

            cell_index = self.calculate_index(particle_cell_location)
            self.particle_neighbour_list[i] = self.cell_list[cell_index]
            self.cell_list[cell_index] = i

    def calculate_index(self, particle_cell_location):
        cell_index = 0
        for k in range(self.dim - 1, -1, -1):
            j = k + 1
            cell_index += int(particle_cell_location[k] * np.prod(self.cell_number[j:3]))
        return cell_index

    def periodic_box_shift(self, particle_pos, box_space):
        while particle_pos >= box_space:
            particle_pos -= box_space
        while particle_pos < 0:
            particle_pos += box_space
        return particle_pos

    def cell_neighbour_list_1D(self):
        cell_nl = np.zeros((3, len(self.cell_list), 2), dtype=np.int32)
        for i in range(int(self.cell_number[0])):
            shift = 0
            sh = np.zeros((1))
            pos = np.array([i], dtype=np.int32)
            cell_index = self.calculate_index(pos)
            cell_nl[0][cell_index][0] = self.calculate_index(pos)
            cell_nl[0][cell_index][1] = shift
            pos = np.array([i + 1], dtype=np.int32)
            for j in range(0, 1):
                [pos[j], sh[j]] = self.cell_shift(j, pos[j])
                shift = sh.sum()
            cell_nl[1][cell_index][0] = self.calculate_index(pos)
            cell_nl[1][cell_index][1] = shift
            pos = np.array([i - 1], dtype=np.int32)
            for j in range(0, 1):
                [pos[j], sh[j]] = self.cell_shift(j, pos[j])
                shift = sh.sum()
            cell_nl[2][cell_index][0] = self.calculate_index(pos)
            cell_nl[2][cell_index][1] = shift
        return cell_nl

    def cell_neighbour_list_2D(self):
        cell_nl = np.zeros((9, len(self.cell_list), 2), dtype=np.int32)
        for i in range(int(self.cell_number[0])):
            for k in range(int(self.cell_number[1])):
                shift = 0
                sh = np.zeros((2))
                pos = np.array([i, k], dtype=np.int32)
                cell_index = self.calculate_index(pos)
                cell_nl[0][cell_index][0] = cell_index
                cell_nl[0][cell_index][1] = shift
                pos = np.array([i + 1, k], dtype=np.int32)
                for j in range(0, 2):
                    [pos[j], sh[j]] = self.cell_shift(j, pos[j])
                    shift = sh.sum()
                cell_nl[1][cell_index][0] = self.calculate_index(pos)
                cell_nl[1][cell_index][1] = shift
                pos = np.array([i - 1, k], dtype=np.int32)
                for j in range(0, 2):
                    [pos[j], sh[j]] = self.cell_shift(j, pos[j])
                    shift = sh.sum()
                cell_nl[2][cell_index][0] = self.calculate_index(pos)
                cell_nl[2][cell_index][1] = shift
                pos = np.array([i + 1, k + 1], dtype=np.int32)
                for j in range(0, 2):
                    [pos[j], sh[j]] = self.cell_shift(j, pos[j])
                    shift = sh.sum()
                cell_nl[3][cell_index][0] = self.calculate_index(pos)
                cell_nl[3][cell_index][1] = shift
                pos = np.array([i - 1, k + 1], dtype=np.int32)
                for j in range(0, 2):
                    [pos[j], sh[j]] = self.cell_shift(j, pos[j])
                    shift = sh.sum()
                cell_nl[4][cell_index][0] = self.calculate_index(pos)
                cell_nl[4][cell_index][1] = shift
                pos = np.array([i + 1, k - 1], dtype=np.int32)
                for j in range(0, 2):
                    [pos[j], sh[j]] = self.cell_shift(j, pos[j])
                    shift = sh.sum()
                cell_nl[5][cell_index][0] = self.calculate_index(pos)
                cell_nl[5][cell_index][1] = shift
                pos = np.array([i - 1, k - 1], dtype=np.int32)
                for j in range(0, 2):
                    [pos[j], sh[j]] = self.cell_shift(j, pos[j])
                    shift = sh.sum()
                cell_nl[6][cell_index][0] = self.calculate_index(pos)
                cell_nl[6][cell_index][1] = shift
                pos = np.array([i, k + 1], dtype=np.int32)
                for j in range(0, 2):
                    [pos[j], sh[j]] = self.cell_shift(j, pos[j])
                    shift = sh.sum()
                cell_nl[7][cell_index][0] = self.calculate_index(pos)
                cell_nl[7][cell_index][1] = shift
                pos = np.array([i, k - 1], dtype=np.int32)
                for j in range(0, 2):
                    [pos[j], sh[j]] = self.cell_shift(j, pos[j])
                    shift = sh.sum()
                cell_nl[8][cell_index][0] = self.calculate_index(pos)
                cell_nl[8][cell_index][1] = shift
        return cell_nl

    def cell_neighbour_list_3D(self):
        cell_nl = np.zeros((27, len(self.cell_list), 2), dtype=np.int32)
        for i in range(int(self.cell_number[0])):
            for k in range(int(self.cell_number[1])):
                for p in range(int(self.cell_number[2])):
                    shift = 0
                    sh = np.zeros((3))
                    pos = np.array([i, k, p], dtype=np.int32)
                    cell_index = self.calculate_index(pos)
                    cell_nl[0][cell_index][0] = cell_index
                    cell_nl[0][cell_index][1] = shift
                    pos = np.array([i + 1, k, p], dtype=np.int32)
                    for j in range(0, 3):
                        [pos[j], sh[j]] = self.cell_shift(j, pos[j])
                        shift = sh.sum()
                    cell_nl[1][cell_index][0] = self.calculate_index(pos)
                    cell_nl[1][cell_index][1] = shift
                    pos = np.array([i - 1, k, p], dtype=np.int32)
                    for j in range(0, 3):
                        [pos[j], sh[j]] = self.cell_shift(j, pos[j])
                        shift = sh.sum()
                    cell_nl[2][cell_index][0] = self.calculate_index(pos)
                    cell_nl[2][cell_index][1] = shift
                    pos = np.array([i + 1, k + 1, p], dtype=np.int32)
                    for j in range(0, 3):
                        [pos[j], sh[j]] = self.cell_shift(j, pos[j])
                        shift = sh.sum()
                    cell_nl[3][cell_index][0] = self.calculate_index(pos)
                    cell_nl[3][cell_index][1] = shift
                    pos = np.array([i - 1, k + 1, p], dtype=np.int32)
                    for j in range(0, 3):
                        [pos[j], sh[j]] = self.cell_shift(j, pos[j])
                        shift = sh.sum()
                    cell_nl[4][cell_index][0] = self.calculate_index(pos)
                    cell_nl[4][cell_index][1] = shift
                    pos = np.array([i + 1, k - 1, p], dtype=np.int32)
                    for j in range(0, 3):
                        [pos[j], sh[j]] = self.cell_shift(j, pos[j])
                        shift = sh.sum()
                    cell_nl[5][cell_index][0] = self.calculate_index(pos)
                    cell_nl[5][cell_index][1] = shift
                    pos = np.array([i - 1, k - 1, p], dtype=np.int32)
                    for j in range(0, 3):
                        [pos[j], sh[j]] = self.cell_shift(j, pos[j])
                        shift = sh.sum()
                    cell_nl[6][cell_index][0] = self.calculate_index(pos)
                    cell_nl[6][cell_index][1] = shift
                    pos = np.array([i, k + 1, p], dtype=np.int32)
                    for j in range(0, 3):
                        [pos[j], sh[j]] = self.cell_shift(j, pos[j])
                        shift = sh.sum()
                    cell_nl[7][cell_index][0] = self.calculate_index(pos)
                    cell_nl[7][cell_index][1] = shift
                    pos = np.array([i, k - 1, p], dtype=np.int32)
                    for j in range(0, 3):
                        [pos[j], sh[j]] = self.cell_shift(j, pos[j])
                        shift = sh.sum()
                    cell_nl[8][cell_index][0] = self.calculate_index(pos)
                    cell_nl[8][cell_index][1] = shift

                    pos = np.array([i, k, p + 1], dtype=np.int32)
                    for j in range(0, 3):
                        [pos[j], sh[j]] = self.cell_shift(j, pos[j])
                        shift = sh.sum()
                    cell_nl[9][cell_index][0] = self.calculate_index(pos)
                    cell_nl[9][cell_index][1] = shift
                    pos = np.array([i + 1, k, p + 1], dtype=np.int32)
                    for j in range(0, 3):
                        [pos[j], sh[j]] = self.cell_shift(j, pos[j])
                        shift = sh.sum()
                    cell_nl[10][cell_index][0] = self.calculate_index(pos)
                    cell_nl[10][cell_index][1] = shift
                    pos = np.array([i - 1, k, p + 1], dtype=np.int32)
                    for j in range(0, 3):
                        [pos[j], sh[j]] = self.cell_shift(j, pos[j])
                        shift = sh.sum()
                    cell_nl[11][cell_index][0] = self.calculate_index(pos)
                    cell_nl[11][cell_index][1] = shift
                    pos = np.array([i + 1, k + 1, p + 1], dtype=np.int32)
                    for j in range(0, 3):
                        [pos[j], sh[j]] = self.cell_shift(j, pos[j])
                        shift = sh.sum()
                    cell_nl[12][cell_index][0] = self.calculate_index(pos)
                    cell_nl[12][cell_index][1] = shift
                    pos = np.array([i - 1, k + 1, p + 1], dtype=np.int32)
                    for j in range(0, 3):
                        [pos[j], sh[j]] = self.cell_shift(j, pos[j])
                        shift = sh.sum()
                    cell_nl[13][cell_index][0] = self.calculate_index(pos)
                    cell_nl[13][cell_index][1] = shift
                    pos = np.array([i + 1, k - 1, p + 1], dtype=np.int32)
                    for j in range(0, 3):
                        [pos[j], sh[j]] = self.cell_shift(j, pos[j])
                        shift = sh.sum()
                    cell_nl[14][cell_index][0] = self.calculate_index(pos)
                    cell_nl[14][cell_index][1] = shift
                    pos = np.array([i - 1, k - 1, p + 1], dtype=np.int32)
                    for j in range(0, 3):
                        [pos[j], sh[j]] = self.cell_shift(j, pos[j])
                        shift = sh.sum()
                    cell_nl[15][cell_index][0] = self.calculate_index(pos)
                    cell_nl[15][cell_index][1] = shift
                    pos = np.array([i, k + 1, p + 1], dtype=np.int32)
                    for j in range(0, 3):
                        [pos[j], sh[j]] = self.cell_shift(j, pos[j])
                        shift = sh.sum()
                    cell_nl[16][cell_index][0] = self.calculate_index(pos)
                    cell_nl[16][cell_index][1] = shift
                    pos = np.array([i, k - 1, p + 1], dtype=np.int32)
                    for j in range(0, 3):
                        [pos[j], sh[j]] = self.cell_shift(j, pos[j])
                        shift = sh.sum()
                    cell_nl[17][cell_index][0] = self.calculate_index(pos)
                    cell_nl[17][cell_index][1] = shift

                    pos = np.array([i, k, p - 1], dtype=np.int32)
                    for j in range(0, 3):
                        [pos[j], sh[j]] = self.cell_shift(j, pos[j])
                        shift = sh.sum()
                    cell_nl[18][cell_index][0] = self.calculate_index(pos)
                    cell_nl[18][cell_index][1] = shift
                    pos = np.array([i + 1, k, p - 1], dtype=np.int32)
                    for j in range(0, 3):
                        [pos[j], sh[j]] = self.cell_shift(j, pos[j])
                        shift = sh.sum()
                    cell_nl[19][cell_index][0] = self.calculate_index(pos)
                    cell_nl[19][cell_index][1] = shift
                    pos = np.array([i - 1, k, p - 1], dtype=np.int32)
                    for j in range(0, 3):
                        [pos[j], sh[j]] = self.cell_shift(j, pos[j])
                        shift = sh.sum()
                    cell_nl[20][cell_index][0] = self.calculate_index(pos)
                    cell_nl[20][cell_index][1] = shift
                    pos = np.array([i + 1, k + 1, p - 1], dtype=np.int32)
                    for j in range(0, 3):
                        [pos[j], sh[j]] = self.cell_shift(j, pos[j])
                        shift = sh.sum()
                    cell_nl[21][cell_index][0] = self.calculate_index(pos)
                    cell_nl[21][cell_index][1] = shift
                    pos = np.array([i - 1, k + 1, p - 1], dtype=np.int32)
                    for j in range(0, 3):
                        [pos[j], sh[j]] = self.cell_shift(j, pos[j])
                        shift = sh.sum()
                    cell_nl[22][cell_index][0] = self.calculate_index(pos)
                    cell_nl[22][cell_index][1] = shift
                    pos = np.array([i + 1, k - 1, p - 1], dtype=np.int32)
                    for j in range(0, 3):
                        [pos[j], sh[j]] = self.cell_shift(j, pos[j])
                        shift = sh.sum()
                    cell_nl[23][cell_index][0] = self.calculate_index(pos)
                    cell_nl[23][cell_index][1] = shift
                    pos = np.array([i - 1, k - 1, p - 1], dtype=np.int32)
                    for j in range(0, 3):
                        [pos[j], sh[j]] = self.cell_shift(j, pos[j])
                        shift = sh.sum()
                    cell_nl[24][cell_index][0] = self.calculate_index(pos)
                    cell_nl[24][cell_index][1] = shift
                    pos = np.array([i, k + 1, p - 1], dtype=np.int32)
                    for j in range(0, 3):
                        [pos[j], sh[j]] = self.cell_shift(j, pos[j])
                        shift = sh.sum()
                    cell_nl[25][cell_index][0] = self.calculate_index(pos)
                    cell_nl[25][cell_index][1] = shift
                    pos = np.array([i, k - 1, p - 1], dtype=np.int32)
                    for j in range(0, 3):
                        [pos[j], sh[j]] = self.cell_shift(j, pos[j])
                        shift = sh.sum()
                    cell_nl[26][cell_index][0] = self.calculate_index(pos)
                    cell_nl[26][cell_index][1] = shift

        return cell_nl

    def cell_shift(self, j, pos):
        k = 0
        if pos < 0:
            pos += self.cell_number[j]
            k = 1
        if pos > self.cell_number[j] - 1:
            pos -= self.cell_number[j]
            k = 1
        return [pos, k]

    def calc_cell_neighbours(self):
        cell_nl = np.zeros((1, 1, 2), dtype=np.int32)
        if self.dim == 1:
            cell_nl = self.cell_neighbour_list_1D()
        if self.dim == 2:
            cell_nl = self.cell_neighbour_list_2D()
        if self.dim == 3:
            cell_nl = self.cell_neighbour_list_3D()
        return cell_nl
