import unittest

from Particle_Simulation.Particle import Particle


class test_Particle(unittest.TestCase):
    def test_wrong_position(self):
        self.assertRaises(TypeError, Particle, 1, 23)
