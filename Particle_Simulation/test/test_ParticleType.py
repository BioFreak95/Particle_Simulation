from Particle_Simulation.ParticleType import ParticleType
import unittest


class test_ParticleType(unittest.TestCase):
    def test_wrong_name(self):
        self.assertRaises(TypeError, ParticleType, 23, 23, 23, 23, 23)

    def test_wrong_mass(self):
        self.assertRaises(ValueError, ParticleType, 'test', -23, 23, 23, 23)

    def test_wrong_charge(self):
        self.assertRaises(ValueError, ParticleType, 'test', 23, -23, 23, 23)

    def test_wrong_ljsigma(self):
        self.assertRaises(ValueError, ParticleType, 'test', 23, 23, -23, 23)

    def test_wrong_ljepsilon(self):
        self.assertRaises(ValueError, ParticleType, 'test', 23, 23, 23, -23)
