import numpy as np
from numba import jitclass
from numba import float32, int8, int32, int16, int64

specs = [
    ('particle_positions', float32[:, :]),
    ('box_space', float32[:]),
    ('cutoff', float32),
    ('particle_number', int32),
    ('dim', int8),
    ('cell_number', int16[:]),
    ('cell_space', float32[:]),
    ('total_cell_number', int32),
    ('cell_list', int64[:]),
    ('particle_neighbour_list', int64[:]),
]


@jitclass(specs)

class Neighbourlist:
    def __init__(self, particles, Box, rc):

        self.particle_positions = particles.astype(np.float32)
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
        self.cell_list = np.zeros(self.total_cell_number, dtype=np.int64) - 1
        self.particle_neighbour_list = np.zeros(self.particle_number, dtype=np.int64) - 1

    def update_neighbourlist(self):
        raise NotImplementedError()

    def construct_neighborlist(self):

        for i in range(self.particle_number):

            particle_cell_location = []
            for a in range(len(self.particle_positions[i])):
                self.particle_positions[i][a] = self.periodic_box_shift(self.particle_positions[i][a],
                                                                        self.box_space[a])
                particle_cell_location.append(np.floor(self.particle_positions[i][a] / self.cell_space[a]))

            cell_index = self.calculate_index(particle_cell_location)
            self.particle_neighbour_list[i] = self.cell_list[cell_index]
            self.cell_list[cell_index] = i

    def calculate_index(self, particle_cell_location):
        cell_index = 0
        k = self.dim - 1
        j = self.dim
        while k >= 0:
            cell_index += int(particle_cell_location[k] * np.prod(self.cell_number[j:3]))
            j = j - 1
            k = k - 1
        return cell_index

    def periodic_box_shift(self, particle_pos, box_space):
        while particle_pos >= box_space:
            particle_pos -= box_space
        while particle_pos < 0:
            particle_pos += box_space
        return particle_pos

    def cell_neighbour_list_1D(self):
        cell_nl = np.zeros((3, len(self.cell_list)), dtype=np.int32)
        for i in range(int(self.cell_number[0])):
            pos = np.array([i], dtype=np.int32)
            cell_index = self.calculate_index(pos)
            cell_nl[0][cell_index] = self.calculate_index(pos)
            pos = np.array([i + 1], dtype=np.int32)
            for j in range(0, 1):
                if pos[j] < 0:
                    pos[j] += self.cell_number[j]
                if pos[j] > self.cell_number[j] - 1:
                    pos[j] -= self.cell_number[j]
            cell_nl[1][cell_index] = self.calculate_index(pos)
            pos = np.array([i - 1], dtype=np.int32)
            for j in range(0, 1):
                if pos[j] < 0:
                    pos[j] += self.cell_number[j]
                if pos[j] > self.cell_number[j] - 1:
                    pos[j] -= self.cell_number[j]
            cell_nl[2][cell_index] = self.calculate_index(pos)
        return cell_nl

    def cell_neighbour_list_2D(self):
        cell_nl = np.zeros((9, len(self.cell_list)), dtype=np.int32)
        for i in range(int(self.cell_number[0])):
            for k in range(int(self.cell_number[1])):
                pos = np.array([i, k], dtype=np.int32)
                cell_index = self.calculate_index(pos)
                cell_nl[0][cell_index] = cell_index
                pos = np.array([i + 1, k], dtype=np.int32)
                for j in range(0, 2):
                    if pos[j] < 0:
                        pos[j] += self.cell_number[j]
                    if pos[j] > self.cell_number[j] - 1:
                        pos[j] -= self.cell_number[j]
                cell_nl[1][cell_index] = self.calculate_index(pos)
                pos = np.array([i - 1, k], dtype=np.int32)
                for j in range(0, 2):
                    if pos[j] < 0:
                        pos[j] += self.cell_number[j]
                    if pos[j] > self.cell_number[j] - 1:
                        pos[j] -= self.cell_number[j]
                cell_nl[2][cell_index] = self.calculate_index(pos)
                pos = np.array([i + 1, k + 1], dtype=np.int32)
                for j in range(0, 2):
                    if pos[j] < 0:
                        pos[j] += self.cell_number[j]
                    if pos[j] > self.cell_number[j] - 1:
                        pos[j] -= self.cell_number[j]
                cell_nl[3][cell_index] = self.calculate_index(pos)
                pos = np.array([i - 1, k + 1], dtype=np.int32)
                for j in range(0, 2):
                    if pos[j] < 0:
                        pos[j] += self.cell_number[j]
                    if pos[j] > self.cell_number[j] - 1:
                        pos[j] -= self.cell_number[j]
                cell_nl[4][cell_index] = self.calculate_index(pos)
                pos = np.array([i + 1, k - 1], dtype=np.int32)
                for j in range(0, 2):
                    if pos[j] < 0:
                        pos[j] += self.cell_number[j]
                    if pos[j] > self.cell_number[j] - 1:
                        pos[j] -= self.cell_number[j]
                cell_nl[5][cell_index] = self.calculate_index(pos)
                pos = np.array([i - 1, k - 1], dtype=np.int32)
                for j in range(0, 2):
                    if pos[j] < 0:
                        pos[j] += self.cell_number[j]
                    if pos[j] > self.cell_number[j] - 1:
                        pos[j] -= self.cell_number[j]
                cell_nl[6][cell_index] = self.calculate_index(pos)
                pos = np.array([i, k + 1], dtype=np.int32)
                for j in range(0, 2):
                    if pos[j] < 0:
                        pos[j] += self.cell_number[j]
                    if pos[j] > self.cell_number[j] - 1:
                        pos[j] -= self.cell_number[j]
                cell_nl[7][cell_index] = self.calculate_index(pos)
                pos = np.array([i, k - 1], dtype=np.int32)
                for j in range(0, 2):
                    if pos[j] < 0:
                        pos[j] += self.cell_number[j]
                    if pos[j] > self.cell_number[j] - 1:
                        pos[j] -= self.cell_number[j]
                cell_nl[8][cell_index] = self.calculate_index(pos)
        return cell_nl

    def cell_neighbour_list_3D(self):
        cell_nl = np.zeros((27, len(self.cell_list)), dtype=np.int32)
        for i in range(int(self.cell_number[0])):
            for k in range(int(self.cell_number[1])):
                for p in range(int(self.cell_number[2])):
                    pos = np.array([i, k, p], dtype=np.int32)
                    cell_index = self.calculate_index(pos)
                    cell_nl[0][cell_index] = cell_index
                    pos = np.array([i + 1, k, p], dtype=np.int32)
                    for j in range(0, 3):
                        if pos[j] < 0:
                            pos[j] += self.cell_number[j]
                        if pos[j] > self.cell_number[j] - 1:
                            pos[j] -= self.cell_number[j]
                    cell_nl[1][cell_index] = self.calculate_index(pos)
                    pos = np.array([i - 1, k, p], dtype=np.int32)
                    for j in range(0, 3):
                        if pos[j] < 0:
                            pos[j] += self.cell_number[j]
                        if pos[j] > self.cell_number[j] - 1:
                            pos[j] -= self.cell_number[j]
                    cell_nl[2][cell_index] = self.calculate_index(pos)
                    pos = np.array([i + 1, k + 1, p], dtype=np.int32)
                    for j in range(0, 3):
                        if pos[j] < 0:
                            pos[j] += self.cell_number[j]
                        if pos[j] > self.cell_number[j] - 1:
                            pos[j] -= self.cell_number[j]
                    cell_nl[3][cell_index] = self.calculate_index(pos)
                    pos = np.array([i - 1, k + 1, p], dtype=np.int32)
                    for j in range(0, 3):
                        if pos[j] < 0:
                            pos[j] += self.cell_number[j]
                        if pos[j] > self.cell_number[j] - 1:
                            pos[j] -= self.cell_number[j]
                    cell_nl[4][cell_index] = self.calculate_index(pos)
                    pos = np.array([i + 1, k - 1, p], dtype=np.int32)
                    for j in range(0, 3):
                        if pos[j] < 0:
                            pos[j] += self.cell_number[j]
                        if pos[j] > self.cell_number[j] - 1:
                            pos[j] -= self.cell_number[j]
                    cell_nl[5][cell_index] = self.calculate_index(pos)
                    pos = np.array([i - 1, k - 1, p], dtype=np.int32)
                    for j in range(0, 3):
                        if pos[j] < 0:
                            pos[j] += self.cell_number[j]
                        if pos[j] > self.cell_number[j] - 1:
                            pos[j] -= self.cell_number[j]
                    cell_nl[6][cell_index] = self.calculate_index(pos)
                    pos = np.array([i, k + 1, p], dtype=np.int32)
                    for j in range(0, 3):
                        if pos[j] < 0:
                            pos[j] += self.cell_number[j]
                        if pos[j] > self.cell_number[j] - 1:
                            pos[j] -= self.cell_number[j]
                    cell_nl[7][cell_index] = self.calculate_index(pos)
                    pos = np.array([i, k - 1, p], dtype=np.int32)
                    for j in range(0, 3):
                        if pos[j] < 0:
                            pos[j] += self.cell_number[j]
                        if pos[j] > self.cell_number[j] - 1:
                            pos[j] -= self.cell_number[j]
                    cell_nl[8][cell_index] = self.calculate_index(pos)

                    pos = np.array([i, k, p + 1], dtype=np.int32)
                    for j in range(0, 3):
                        if pos[j] < 0:
                            pos[j] += self.cell_number[j]
                        if pos[j] > self.cell_number[j] - 1:
                            pos[j] -= self.cell_number[j]
                    cell_nl[9][cell_index] = self.calculate_index(pos)
                    pos = np.array([i + 1, k, p + 1], dtype=np.int32)
                    for j in range(0, 3):
                        if pos[j] < 0:
                            pos[j] += self.cell_number[j]
                        if pos[j] > self.cell_number[j] - 1:
                            pos[j] -= self.cell_number[j]
                    cell_nl[10][cell_index] = self.calculate_index(pos)
                    pos = np.array([i - 1, k, p + 1], dtype=np.int32)
                    for j in range(0, 3):
                        if pos[j] < 0:
                            pos[j] += self.cell_number[j]
                        if pos[j] > self.cell_number[j] - 1:
                            pos[j] -= self.cell_number[j]
                    cell_nl[11][cell_index] = self.calculate_index(pos)
                    pos = np.array([i + 1, k + 1, p + 1], dtype=np.int32)
                    for j in range(0, 3):
                        if pos[j] < 0:
                            pos[j] += self.cell_number[j]
                        if pos[j] > self.cell_number[j] - 1:
                            pos[j] -= self.cell_number[j]
                    cell_nl[12][cell_index] = self.calculate_index(pos)
                    pos = np.array([i - 1, k + 1, p + 1], dtype=np.int32)
                    for j in range(0, 3):
                        if pos[j] < 0:
                            pos[j] += self.cell_number[j]
                        if pos[j] > self.cell_number[j] - 1:
                            pos[j] -= self.cell_number[j]
                    cell_nl[13][cell_index] = self.calculate_index(pos)
                    pos = np.array([i + 1, k - 1, p + 1], dtype=np.int32)
                    for j in range(0, 3):
                        if pos[j] < 0:
                            pos[j] += self.cell_number[j]
                        if pos[j] > self.cell_number[j] - 1:
                            pos[j] -= self.cell_number[j]
                    cell_nl[14][cell_index] = self.calculate_index(pos)
                    pos = np.array([i - 1, k - 1, p + 1], dtype=np.int32)
                    for j in range(0, 3):
                        if pos[j] < 0:
                            pos[j] += self.cell_number[j]
                        if pos[j] > self.cell_number[j] - 1:
                            pos[j] -= self.cell_number[j]
                    cell_nl[15][cell_index] = self.calculate_index(pos)
                    pos = np.array([i, k + 1, p + 1], dtype=np.int32)
                    for j in range(0, 3):
                        if pos[j] < 0:
                            pos[j] += self.cell_number[j]
                        if pos[j] > self.cell_number[j] - 1:
                            pos[j] -= self.cell_number[j]
                    cell_nl[16][cell_index] = self.calculate_index(pos)
                    pos = np.array([i, k - 1, p + 1], dtype=np.int32)
                    for j in range(0, 3):
                        if pos[j] < 0:
                            pos[j] += self.cell_number[j]
                        if pos[j] > self.cell_number[j] - 1:
                            pos[j] -= self.cell_number[j]
                    cell_nl[17][cell_index] = self.calculate_index(pos)

                    pos = np.array([i, k, p - 1], dtype=np.int32)
                    for j in range(0, 3):
                        if pos[j] < 0:
                            pos[j] += self.cell_number[j]
                        if pos[j] > self.cell_number[j] - 1:
                            pos[j] -= self.cell_number[j]
                    cell_nl[18][cell_index] = self.calculate_index(pos)
                    pos = np.array([i + 1, k, p - 1], dtype=np.int32)
                    for j in range(0, 3):
                        if pos[j] < 0:
                            pos[j] += self.cell_number[j]
                        if pos[j] > self.cell_number[j] - 1:
                            pos[j] -= self.cell_number[j]
                    cell_nl[19][cell_index] = self.calculate_index(pos)
                    pos = np.array([i - 1, k, p - 1], dtype=np.int32)
                    for j in range(0, 3):
                        if pos[j] < 0:
                            pos[j] += self.cell_number[j]
                        if pos[j] > self.cell_number[j] - 1:
                            pos[j] -= self.cell_number[j]
                    cell_nl[20][cell_index] = self.calculate_index(pos)
                    pos = np.array([i + 1, k + 1, p - 1], dtype=np.int32)
                    for j in range(0, 3):
                        if pos[j] < 0:
                            pos[j] += self.cell_number[j]
                        if pos[j] > self.cell_number[j] - 1:
                            pos[j] -= self.cell_number[j]
                    cell_nl[21][cell_index] = self.calculate_index(pos)
                    pos = np.array([i - 1, k + 1, p - 1], dtype=np.int32)
                    for j in range(0, 3):
                        if pos[j] < 0:
                            pos[j] += self.cell_number[j]
                        if pos[j] > self.cell_number[j] - 1:
                            pos[j] -= self.cell_number[j]
                    cell_nl[22][cell_index] = self.calculate_index(pos)
                    pos = np.array([i + 1, k - 1, p - 1], dtype=np.int32)
                    for j in range(0, 3):
                        if pos[j] < 0:
                            pos[j] += self.cell_number[j]
                        if pos[j] > self.cell_number[j] - 1:
                            pos[j] -= self.cell_number[j]
                    cell_nl[23][cell_index] = self.calculate_index(pos)
                    pos = np.array([i - 1, k - 1, p - 1], dtype=np.int32)
                    for j in range(0, 3):
                        if pos[j] < 0:
                            pos[j] += self.cell_number[j]
                        if pos[j] > self.cell_number[j] - 1:
                            pos[j] -= self.cell_number[j]
                    cell_nl[24][cell_index] = self.calculate_index(pos)
                    pos = np.array([i, k + 1, p - 1], dtype=np.int32)
                    for j in range(0, 3):
                        if pos[j] < 0:
                            pos[j] += self.cell_number[j]
                        if pos[j] > self.cell_number[j] - 1:
                            pos[j] -= self.cell_number[j]
                    cell_nl[25][cell_index] = self.calculate_index(pos)
                    pos = np.array([i, k - 1, p - 1], dtype=np.int32)
                    for j in range(0, 3):
                        if pos[j] < 0:
                            pos[j] += self.cell_number[j]
                        if pos[j] > self.cell_number[j] - 1:
                            pos[j] -= self.cell_number[j]
                    cell_nl[26][cell_index] = self.calculate_index(pos)

        return cell_nl

    def calc_cell_neighbours(self):
        cell_nl = np.zeros((1, 1))
        if self.dim == 1:
            cell_nl = self.cell_neighbour_list_1D()
        if self.dim == 2:
            cell_nl = self.cell_neighbour_list_2D()
        if self.dim == 3:
            cell_nl = self.cell_neighbour_list_3D()
        return cell_nl


'''
    Old Code - Don't delete till v1.0

    def construct_3dneighborlist(self):
        self.cellnumber = np.floor(np.array([self.Box[0]/self.rc,self.Box[1]/self.rc,self.Box[2]/self.rc]))
        self.cellspace = np.array([self.Box[0]/self.cellnumber[0],self.Box[1]/self.cellnumber[1],
                                   self.Box[2]/self.cellnumber[2]])
        self.totalcellnumber = int(np.prod(self.cellnumber))
        self.head = np.zeros(self.totalcellnumber)-1
        self.list = np.zeros(self.particlenumber)-1
        self.mc = np.zeros(3)
    
        for i in range(self.particlenumber):
            for a in range(0,2):
                self.mc[a] = np.floor(self.particles[i][a] / self.cellspace[a])
            index = int(self.mc[2] + self.mc[1]*self.cellnumber[2] + self.mc[0]*self.cellnumber[1]*self.cellnumber[2])
            self.list[i] = self.head[index]
            self.head[index] = i
    
    self.cell_number = np.ones(3)  
    self.cell_space = np.zeros(3)

    cell_index = int(particle_cell_location[2] + particle_cell_location[1] * self.cell_number[2] +
        particle_cell_location[0] * self.cell_number[2] * self.cell_number[1])
'''
