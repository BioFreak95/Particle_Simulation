import unittest
import numpy as np
from Particle_Simulation.EnergyCalculator import EnergyCalculator
from Particle_Simulation.Neighbourlist import Neighbourlist


class test_EnergyCalculator(unittest.TestCase):
    def test__calculate_dot_product(self):
        vector1 = np.array([2, 3, 4])
        vector2 = np.array([3, 1, 6])
        reference_dot = 33
        EC = EnergyCalculator(box=np.array([1.]), cutoff_radius=1, es_sigma=1, charges=np.array([1]),
                              lj_sigmas=np.array([1]), lj_epsilons=np.array([1]), k_vector=np.array([[1]]))
        calc = EC._calculate_dot_product(vector1, vector2)
        self.assertEqual(calc, reference_dot)

    def test_wrap_distance(self):
        EC = EnergyCalculator(box=np.array([2, 2, 2]), cutoff_radius=1, es_sigma=1, charges=np.array([1]),
                              lj_sigmas=np.array([1]), lj_epsilons=np.array([1]), k_vector=np.array([[1]]))
        vector1 = np.array([1, -2, 2])
        calc_wrap = EC._wrap_distance(vector1)
        reference = np.array([-1, 0, 0])
        self.assertTrue(np.array_equal(calc_wrap, reference))

    def test_determine_box_shift(self):
        EC = EnergyCalculator(box=np.array([2, 3, 2]), cutoff_radius=1, es_sigma=1, charges=np.array([1]),
                              lj_sigmas=np.array([1]), lj_epsilons=np.array([1]), k_vector=np.array([[1]]))
        vector1 = np.array([3, 4, 1])
        NL = Neighbourlist(Box=np.array([2, 3, 2]), particles=np.array([[3, 4, 1]]), rc=1)
        cell_nl = NL.cell_neighbour_list_3D()
        EC.cell_neighbour_list = cell_nl
        calc_shift = EC._determine_box_shift(0, 2)
        reference_shift = np.array([-2, 0, 0])
        self.assertTrue(np.array_equal(calc_shift, reference_shift))

    def test_calculate_norm(self):
        EC = EnergyCalculator(box=np.array([1.]), cutoff_radius=1, es_sigma=1, charges=np.array([1]),
                              lj_sigmas=np.array([1, 2, 3, 4, 5, 6, 7, 8]),
                              lj_epsilons=np.array([1, 2, 3, 4, 5, 6, 7, 8]), k_vector=np.array([[1]]))
        vector1 = np.array([2, 3, 4])
        vector2 = np.array([3, 1, 6])
        reference_norm = np.sqrt(9)
        calc_norm = EC._calculate_norm((vector2 - vector1))
        self.assertEqual(reference_norm, calc_norm)

    def test_determine_epsilon(self):
        EC = EnergyCalculator(box=np.array([1]), cutoff_radius=1, es_sigma=1, charges=np.array([1]),
                              lj_sigmas=np.array([1, 2, 3, 4, 5, 6, 7, 8]),
                              lj_epsilons=np.array([1, 2, 3, 4, 5, 6, 7, 8]), k_vector=np.array([[1]]))
        det_epsilon = EC._determine_epsilon(4, 5)
        reference = np.sqrt(30)
        self.assertEqual(det_epsilon, reference)

    def test_determine_sigma(self):
        EC = EnergyCalculator(box=np.array([1.]), cutoff_radius=1, es_sigma=1, charges=np.array([1]),
                              lj_sigmas=np.array([1, 2, 3, 4, 5, 6, 7, 8]),
                              lj_epsilons=np.array([1, 2, 3, 4, 5, 6, 7, 8]), k_vector=np.array([[1]]))

        det_sigma = EC._determine_sigma(4, 5)
        reference = 5.5
        self.assertEqual(det_sigma, reference)
