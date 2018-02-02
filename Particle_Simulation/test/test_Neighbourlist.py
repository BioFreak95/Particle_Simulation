import numpy as np
import unittest
import numpy.testing as npt
from Particle_Simulation.Neighbourlist import Neighbourlist


class test_System(unittest.TestCase):
    # Cell Neighbours Matrix
    def test_cnlist1d_1(self):
        particle_positions = np.array([[9], [4], [1]])
        box_space = np.array([12])
        cutoff = 3
        s1 = Neighbourlist(particle_positions, box_space, cutoff)
        print(s1.calculate_index(np.array([1, 0, 0])))
        reference_CnList = np.array([
            [3, 0, 1, 2],
            [0, 1, 2, 3],
            [1, 2, 3, 0]
        ])
        cnlist = s1.cell_neighbour_list_1D()
        print(cnlist)
        sorted_referenceCnList = np.sort(reference_CnList, axis=0)
        sorted_outputCnList = np.sort(cnlist[:, :, 0], axis=0)
        npt.assert_equal(sorted_referenceCnList, sorted_outputCnList, 'Failed', verbose=True)

    def test_cnlist2d_1(self):
        particle_positions = np.array([[9, 6], [4, 5], [1, 3]])
        box_space = np.array([12, 13])
        cutoff = 3
        s1 = Neighbourlist(particle_positions, box_space, cutoff)
        reference_CnList = np.array([
            [13, 14, 15, 12, 1, 2, 3, 0, 5, 6, 7, 4, 9, 10, 11, 8],
            [1, 2, 3, 0, 5, 6, 7, 4, 9, 10, 11, 8, 13, 14, 15, 12],
            [5, 6, 7, 4, 9, 10, 11, 8, 13, 14, 15, 12, 1, 2, 3, 0],
            [12, 13, 14, 15, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11],
            [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15],
            [4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 0, 1, 2, 3],
            [15, 12, 13, 14, 3, 0, 1, 2, 7, 4, 5, 6, 11, 8, 9, 10],
            [3, 0, 1, 2, 7, 4, 5, 6, 11, 8, 9, 10, 15, 12, 13, 14],
            [7, 4, 5, 6, 11, 8, 9, 10, 15, 12, 13, 14, 3, 0, 1, 2]
        ])
        cnlist = s1.cell_neighbour_list_2D()
        sorted_referenceCnList = np.sort(reference_CnList, axis=0)
        sorted_outputCnList = np.sort(cnlist[:, :, 0], axis=0)
        npt.assert_equal(sorted_referenceCnList, sorted_outputCnList, 'Failed', verbose=True)

    def test_cnlist3d_2(self):
        particle_positions = np.array([[9, 6, 4], [4, 5, 6], [1, 3, 6]])
        box_space = np.array([12, 10, 10])
        cutoff = 3
        s1 = Neighbourlist(particle_positions, box_space, cutoff)
        reference_CnList = np.array([
            [31, 32, 30, 34, 35, 33, 28, 29, 27, 4, 5, 3, 7, 8, 6, 1, 2, 0, 13, 14, 12, 16, 17, 15, 10, 11, 9, 22, 23,
             21, 25, 26, 24, 19, 20, 18],
            [4, 5, 3, 7, 8, 6, 1, 2, 0, 13, 14, 12, 16, 17, 15, 10, 11, 9, 22, 23, 21, 25, 26, 24, 19, 20, 18, 31, 32,
             30, 34, 35, 33, 28, 29, 27],
            [13, 14, 12, 16, 17, 15, 10, 11, 9, 22, 23, 21, 25, 26, 24, 19, 20, 18, 31, 32, 30, 34, 35, 33, 28, 29, 27,
             4, 5, 3, 7, 8, 6, 1, 2, 0],
            [28, 29, 27, 31, 32, 30, 34, 35, 33, 1, 2, 0, 4, 5, 3, 7, 8, 6, 10, 11, 9, 13, 14, 12, 16, 17, 15, 19, 20,
             18, 22, 23, 21, 25, 26, 24],
            [1, 2, 0, 4, 5, 3, 7, 8, 6, 10, 11, 9, 13, 14, 12, 16, 17, 15, 19, 20, 18, 22, 23, 21, 25, 26, 24, 28, 29,
             27, 31, 32, 30, 34, 35, 33],
            [10, 11, 9, 13, 14, 12, 16, 17, 15, 19, 20, 18, 22, 23, 21, 25, 26, 24, 28, 29, 27, 31, 32, 30, 34, 35, 33,
             1, 2, 0, 4, 5, 3, 7, 8, 6],
            [34, 35, 33, 28, 29, 27, 31, 32, 30, 7, 8, 6, 1, 2, 0, 4, 5, 3, 16, 17, 15, 10, 11, 9, 13, 14, 12, 25, 26,
             24, 19, 20, 18, 22, 23, 21],
            [7, 8, 6, 1, 2, 0, 4, 5, 3, 16, 17, 15, 10, 11, 9, 13, 14, 12, 25, 26, 24, 19, 20, 18, 22, 23, 21, 34, 35,
             33, 28, 29, 27, 31, 32, 30],
            [16, 17, 15, 10, 11, 9, 13, 14, 12, 25, 26, 24, 19, 20, 18, 22, 23, 21, 34, 35, 33, 28, 29, 27, 31, 32, 30,
             7, 8, 6, 1, 2, 0, 4, 5, 3],
            [30, 31, 32, 33, 34, 35, 27, 28, 29, 3, 4, 5, 6, 7, 8, 0, 1, 2, 12, 13, 14, 15, 16, 17, 9, 10, 11, 21, 22,
             23, 24, 25, 26, 18, 19, 20],
            [3, 4, 5, 6, 7, 8, 0, 1, 2, 12, 13, 14, 15, 16, 17, 9, 10, 11, 21, 22, 23, 24, 25, 26, 18, 19, 20, 30, 31,
             32, 33, 34, 35, 27, 28, 29],
            [12, 13, 14, 15, 16, 17, 9, 10, 11, 21, 22, 23, 24, 25, 26, 18, 19, 20, 30, 31, 32, 33, 34, 35, 27, 28, 29,
             3, 4, 5, 6, 7, 8, 0, 1, 2],
            [27, 28, 29, 30, 31, 32, 33, 34, 35, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19,
             20, 21, 22, 23, 24, 25, 26],
            [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28,
             29, 30, 31, 32, 33, 34, 35],
            [9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35,
             0, 1, 2, 3, 4, 5, 6, 7, 8],
            [33, 34, 35, 27, 28, 29, 30, 31, 32, 6, 7, 8, 0, 1, 2, 3, 4, 5, 15, 16, 17, 9, 10, 11, 12, 13, 14, 24, 25,
             26, 18, 19, 20, 21, 22, 23],
            [6, 7, 8, 0, 1, 2, 3, 4, 5, 15, 16, 17, 9, 10, 11, 12, 13, 14, 24, 25, 26, 18, 19, 20, 21, 22, 23, 33, 34,
             35, 27, 28, 29, 30, 31, 32],
            [15, 16, 17, 9, 10, 11, 12, 13, 14, 24, 25, 26, 18, 19, 20, 21, 22, 23, 33, 34, 35, 27, 28, 29, 30, 31, 32,
             6, 7, 8, 0, 1, 2, 3, 4, 5],
            [32, 30, 31, 35, 33, 34, 29, 27, 28, 5, 3, 4, 8, 6, 7, 2, 0, 1, 14, 12, 13, 17, 15, 16, 11, 9, 10, 23, 21,
             22, 26, 24, 25, 20, 18, 19],
            [5, 3, 4, 8, 6, 7, 2, 0, 1, 14, 12, 13, 17, 15, 16, 11, 9, 10, 23, 21, 22, 26, 24, 25, 20, 18, 19, 32, 30,
             31, 35, 33, 34, 29, 27, 28],
            [14, 12, 13, 17, 15, 16, 11, 9, 10, 23, 21, 22, 26, 24, 25, 20, 18, 19, 32, 30, 31, 35, 33, 34, 29, 27, 28,
             5, 3, 4, 8, 6, 7, 2, 0, 1],
            [29, 27, 28, 32, 30, 31, 35, 33, 34, 2, 0, 1, 5, 3, 4, 8, 6, 7, 11, 9, 10, 14, 12, 13, 17, 15, 16, 20, 18,
             19, 23, 21, 22, 26, 24, 25],
            [2, 0, 1, 5, 3, 4, 8, 6, 7, 11, 9, 10, 14, 12, 13, 17, 15, 16, 20, 18, 19, 23, 21, 22, 26, 24, 25, 29, 27,
             28, 32, 30, 31, 35, 33, 34],
            [11, 9, 10, 14, 12, 13, 17, 15, 16, 20, 18, 19, 23, 21, 22, 26, 24, 25, 29, 27, 28, 32, 30, 31, 35, 33, 34,
             2, 0, 1, 5, 3, 4, 8, 6, 7],
            [35, 33, 34, 29, 27, 28, 32, 30, 31, 8, 6, 7, 2, 0, 1, 5, 3, 4, 17, 15, 16, 11, 9, 10, 14, 12, 13, 26, 24,
             25, 20, 18, 19, 23, 21, 22],
            [8, 6, 7, 2, 0, 1, 5, 3, 4, 17, 15, 16, 11, 9, 10, 14, 12, 13, 26, 24, 25, 20, 18, 19, 23, 21, 22, 35, 33,
             34, 29, 27, 28, 32, 30, 31],
            [17, 15, 16, 11, 9, 10, 14, 12, 13, 26, 24, 25, 20, 18, 19, 23, 21, 22, 35, 33, 34, 29, 27, 28, 32, 30, 31,
             8, 6, 7, 2, 0, 1, 5, 3, 4]
        ])
        cnlist = s1.cell_neighbour_list_3D()
        print(cnlist)
        sorted_referenceCnList = np.sort(reference_CnList, axis=0)
        sorted_outputCnList = np.sort(cnlist[:, :, 0], axis=0)
        npt.assert_equal(sorted_referenceCnList, sorted_outputCnList, 'Failed', verbose=True)

    # Output Validation

    def test_1d_output_validation_1(self):
        particle_positions = np.array([[1.5], [2.5],
                                       [3.0], [4.5], [5.5],
                                       [6.0], [6.5], [7.0],
                                       [8.25], [9.75], [10.0]])
        reference_head = np.asarray([10.0, 6.0, 9.0])
        reference_neighlist = np.asarray([-1.0, 0.0, 1.0, -1.0, 3.0, 4.0, 5.0, -1.0, 7.0, 8.0, 2.0])
        box_space = np.array([10])
        cutoff = 3
        s1 = Neighbourlist(particle_positions, box_space, cutoff)
        npt.assert_equal(reference_head, s1.cell_list, 'Failed', verbose=True)
        npt.assert_equal(reference_neighlist, s1.particle_neighbour_list, 'Failed', verbose=True)

    def test_1d_output_validation_2(self):
        particle_positions = np.array([[1.5], [2.5],
                                       [3.0], [4.5], [5.5],
                                       [6.0], [6.5], [7.0],
                                       [8.25], [9.75], [10.0], [11.5], [12.5]])
        reference_head = np.asarray([12.0, 6.0, 9.0])
        reference_neighlist = np.asarray([-1.0, 0.0, 1.0, -1.0, 3.0, 4.0, 5.0, -1.0, 7.0, 8.0, 2.0, 10.0, 11.0])
        box_space = np.array([10])
        cutoff = 3
        s1 = Neighbourlist(particle_positions, box_space, cutoff)
        npt.assert_equal(reference_head, s1.cell_list, 'Failed', verbose=True)
        npt.assert_equal(reference_neighlist, s1.particle_neighbour_list, 'Failed', verbose=True)

    def test_1d_output_validation_3(self):
        particle_positions = np.array([[-1.75], [-0.25], [1.5], [2.5],
                                       [3.0], [4.5], [5.5],
                                       [6.0], [6.5], [7.0],
                                       [8.25], [9.75], [10.0], [11.5], [12.5]])
        reference_head = np.asarray([14.0, 8.0, 11.0])
        reference_neighlist = np.asarray(
            [-1.0, 0.0, -1.0, 2.0, 3.0, -1.0, 5.0, 6.0, 7.0, 1.0, 9.0, 10.0, 4.0, 12.0, 13.0])
        box_space = np.array([10])
        cutoff = 3
        s1 = Neighbourlist(particle_positions, box_space, cutoff)
        npt.assert_equal(reference_head, s1.cell_list, 'Failed', verbose=True)
        npt.assert_equal(reference_neighlist, s1.particle_neighbour_list, 'Failed', verbose=True)

    def test_2d_output_validation_1(self):
        particle_positions = np.array([[1.5, 2.], [2, 5.], [2.5, 8.],
                                       [4., 4.], [4., 8.], [5.5, 9.],
                                       [8., 11.], [7., 2.], [6., 5.],
                                       [6., 6.5], [7., 6.5], [10., 2.],
                                       [11., 9.], [11., 11.], [13.5, 2.],
                                       [14., 5.], [-2., 2.], [25.5, 2.],
                                       [-14., 2.], [13.5, 15.], [-1., -2.]])
        box_space = np.array([12., 13.])
        cutoff = 3
        reference_head = np.asarray([19, 15, 2, -1, -1, 3, 5, -1, 7, 8, 10, 6, 18, -1, 12, 20])
        reference_neighlist = np.asarray([-1, -1, -1, -1, -1, 4, -1, -1, -1, -1, 9,
                                          -1, -1, -1, 0, 1, 11, 14, 16, 17, 13])
        s1 = Neighbourlist(particle_positions, box_space, cutoff)
        npt.assert_equal(reference_head, s1.cell_list, 'Failed', verbose=True)
        npt.assert_equal(reference_neighlist, s1.particle_neighbour_list, 'Failed', verbose=True)

    def test_2d_output_validation_2(self):
        particle_positions = np.array([[13.5, 2.], [14., 5.], [-2., 2.], [25.5, 2.],
                                       [-14., 2.], [13.5, 15.], [-1., -2.]])
        box_space = np.array([12., 13.])
        cutoff = 3
        reference_head = np.asarray([5, 1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 4, -1, -1, 6])
        reference_neighlist = np.asarray([-1, -1, -1, 0, 2, 3, -1])
        s1 = Neighbourlist(particle_positions, box_space, cutoff)
        npt.assert_equal(reference_head, s1.cell_list, 'Failed', verbose=True)
        npt.assert_equal(reference_neighlist, s1.particle_neighbour_list, 'Failed', verbose=True)

    def test_3d_output_validation(self):
        particle_positions = np.array([[1.5, 1., 2.5], [1.5, 4., 2.5], [1.5, 8.5, 2.5],
                                       [1.5, 1., 5.5], [1.5, 4., 5.5], [1.5, 5.5, 5.5],
                                       [1.5, 10., 5.5], [1.5, 1., 8.5], [1.5, 4., 8.5],
                                       [1.5, 8.5, 8.5], [1.5, 4., 11.5], [1.5, 11.5, 11.5],
                                       [4.5, 1., 2.5], [4.5, 5.5, 2.5], [4.5, 7., 2.5],
                                       [4.5, 1., 5.5], [4.5, 4., 5.5], [4.5, 5.5, 5.5],
                                       [4.5, 7., 5.5], [4.5, 10., 5.5], [4.5, 2.5, 8.5],
                                       [4.5, 4., 8.5], [4.5, 5.5, 8.5], [4.5, 8.5, 8.5],
                                       [4.5, 1., 11.5], [4.5, 5.5, 11.5], [7.5, 1., 2.5],
                                       [7.5, 5.5, 2.5], [7.5, 7., 2.5], [7.5, 11.5, 2.5],
                                       [7.5, 1., 5.5], [7.5, 4., 5.5], [7.5, 5.5, 5.5],
                                       [7.5, 10., 5.5], [7.5, 2.5, 8.5], [7.5, 5.5, 8.5],
                                       [7.5, 7., 8.5], [7.5, 11.5, 8.5], [7.5, 4., 11.5],
                                       [7.5, 7., 11.5], [10.5, 1., 2.5], [10.5, 5.5, 2.5],
                                       [10.5, 11.5, 2.5], [10.5, 1., 5.5], [10.5, 2.5, 5.5],
                                       [10.5, 4., 5.5], [10.5, 10., 5.5], [10.5, 11.5, 5.5],
                                       [10.5, 1., 8.5], [10.5, 4., 8.5], [10.5, 11.5, 8.5],
                                       [10.5, 1., 11.5], [10.5, 4., 11.5], [10.5, 11.5, 11.5],
                                       [3., 3.25, 3.5], [6.0, 6.5, 7.0], [6.5, 5.0, 7.0],
                                       [8.0, 6.5, 7.0], [12.0, 6.5, 7.0], [12.0, 6.5, 3.5],
                                       [12.0, 13.0, 7.0], [6.0, 13.0, 7.0], [13.5, 1., 5.5],
                                       [13.5, 4., 5.5], [13.5, 5.5, 5.5], [13.5, 10., 5.5],
                                       [13.5, 14, 5.5], [13.5, 15.5, 5.5], [-1.5, 1., 5.5],
                                       [-1.5, 2.5, 5.5], [-1.5, 4., 5.5], [-1.5, 10., 5.5],
                                       [-1.5, 11.5, 5.5]])

        box_space = np.array([12., 13., 14.])
        reference_head = np.asarray([0, 67, 60, -1, 1, 64, 8, 10, 2, 59, 58, -1, -1, 65, -1, 11,
                                     12, 15, 20, 24, 13, 54, 22, 25, 14, 18, 23, -1, -1, 19, -1,
                                     - 1, 26, 30, 61, -1, 27, 32, 56, 38, 28, -1, 57, 39, 29, 33,
                                     37, -1, 40, 69, 48, 51, 41, 70, 49, 52, -1, -1, -1, -1, 42,
                                     72, 50, 53])
        reference_neighlist = np.asarray([-1, -1, -1, -1, -1, 4, -1, -1, -1, -1, -1, -1, -1, -1,
                                          - 1, -1, -1, 16, -1, -1, -1, -1, 21, -1, -1, -1, -1, -1,
                                          - 1, -1, -1, -1, 31, -1, -1, -1, -1, -1, -1, -1, -1, -1,
                                          - 1, -1, 43, -1, -1, 46, -1, -1, -1, -1, -1, -1, 17, 36,
                                          35, 55, 9, -1, 7, 34, 3, 5, 63, 6, 62, 66, 44, 68, 45,
                                          47, 71, ])

        cutoff = 3
        s1 = Neighbourlist(particle_positions, box_space, cutoff)
        npt.assert_equal(reference_head, s1.cell_list, 'Failed', verbose=True)
        npt.assert_equal(reference_neighlist, s1.particle_neighbour_list, 'Failed', verbose=True)


'''
    def test_benchmark(self):
        particle_positions = []
        for i in range(0, 100000):
            xtest = np.random.rand(3)
            particle_positions.append(xtest)
        particle_positions = np.array(particle_positions)
        box_space = np.array([50, 50, 50])
        cutoff = 2
        s1 = Neighbourlist(particle_positions, box_space, cutoff)
        print("rdy")
        for i in range(10):
            s1.construct_neighbourlist()
        print(s1.cell_list)
        print(s1.particle_neighbour_list)

Benchmark:

First step! High Initialisation Time.
1000 steps, 25x25x25 cells and 1000 particels need bit more then 1 second->

10.000.000 Particles 15 seconds calculating time and 1 GB RAM for head and list
15.625.000 cells (250x250x250) 90 seconds calculating time for cell_nl and 4 GB for matrix

both 105 seconds and 4.1 GB RAM

'''
