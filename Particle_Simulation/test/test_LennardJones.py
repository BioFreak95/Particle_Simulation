import numpy as np
import numpy.testing as npt
import unittest
from Particle_Simulation.Particle import Particle
from Particle_Simulation.Parameters import Parameters
from Particle_Simulation.LennardJones import LennardJones
from Particle_Simulation.ParticleType import ParticleType


class test_LennardJones(unittest.TestCase):
    def test_calculate_distance_1(self):
        reference_distance = 11.0
        particle_1 = Particle(position=np.array([1, 0, 0]), type_index=1)
        particle_2 = Particle(position=np.array([12, 0, 0]), type_index=1)
        distance = LennardJones._calculate_distance(particle_1, particle_2)
        npt.assert_equal(reference_distance, distance, 'Failed', verbose=True)

    def test_calculate_distance_2(self):
        reference_distance = 3.5
        particle_1 = Particle(position=np.array([1, 2, 3]), type_index=1)
        particle_2 = Particle(position=np.array([2, 3.5, 6]), type_index=1)
        distance = LennardJones._calculate_distance(particle_1, particle_2)
        npt.assert_equal(reference_distance, distance, 'Failed', verbose=True)

    def test_calculate_lennardjones_potential(self):
        reference_potential = -0.000042499
        print(reference_potential)
        particle_type = ParticleType(name="Natrium", mass=2, charge=2, lj_epsilon=1.25, lj_sigma=0.5)
        particle_type = np.array([particle_type])
        parameters = Parameters(temperature=0, es_sigma=0, mc_update_radius=1, particle_types=particle_type)
        particle_1 = Particle(position=np.array([1, 2, 3]), type_index=0)
        particle_2 = Particle(position=np.array([2, 3.5, 6]), type_index=0)
        lg_value = LennardJones._calculate_potential(particle_1, particle_2, parameters)
        lg_value_rounded = lg_value.round(decimals=9)
        print(lg_value_rounded)
        npt.assert_equal(reference_potential, lg_value_rounded, 'Failed', verbose=True)


if __name__ == '__main__':
    unittest.main()
