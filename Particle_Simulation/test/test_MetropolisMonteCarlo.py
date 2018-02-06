import numpy as np
import numpy.testing as npt
import unittest

from Particle_Simulation.Particle import Particle
from Particle_Simulation.System import System
from Particle_Simulation.Parameters import Parameters
from Particle_Simulation.MetropolisMonteCarlo import MetropolisMonteCarlo


class test_MetropolisMonteCarlo(unittest.TestCase):

    def test_evaluate_trial_configuration_greedy_1(self):

        # set test parameters
        charges = np.ones(10).astype(np.float32)
        lj_sigmas = np.ones(10).astype(np.float32)
        lj_epsilons = np.ones(10).astype(np.float32)

        para = Parameters(temperature=1, box=np.array([1,1,1]), es_sigma=1, cutoff_radius=1, update_radius=1,
                          K_cutoff=1, charges=charges, lj_sigmas=lj_sigmas, lj_epsilons=lj_epsilons,
                          update_probability=0.5)

        particle1 = Particle(np.array([0.5, 0.5, 0.5]))
        particle2 = Particle(np.array([0.5, 0.5, 0.1]))
        particles = [particle1, particle2]

        system = System(particles=particles, parameters=para)
        system.energy.overall_energy = 1
        trial_system = System(particles=particles, parameters=para)
        trial_system.energy.overall_energy = 1.1

        actual = MetropolisMonteCarlo.evaluate_trial_configuration_greedy(system,trial_system)

        np.array_equal(actual, system)

    def test_evaluate_trial_configuration_greedy_2(self):

        # set test parameters
        charges = np.ones(10).astype(np.float32)
        lj_sigmas = np.ones(10).astype(np.float32)
        lj_epsilons = np.ones(10).astype(np.float32)

        para = Parameters(temperature=1, box=np.array([1,1,1]), es_sigma=1, cutoff_radius=1, update_radius=1,
                          K_cutoff=1, charges=charges, lj_sigmas=lj_sigmas, lj_epsilons=lj_epsilons,
                          update_probability=0.5)

        particle1 = Particle(np.array([0.5, 0.5, 0.5]))
        particle2 = Particle(np.array([0.5, 0.5, 0.1]))
        particles = [particle1, particle2]

        system = System(particles=particles, parameters=para)
        system.energy.overall_energy = 1.1
        trial_system = System(particles=particles, parameters=para)
        trial_system.energy.overall_energy = 1.1

        actual = MetropolisMonteCarlo.evaluate_trial_configuration_greedy(system,trial_system)

        np.array_equal(actual, trial_system)

    def test_evaluate_trial_configuration_greedy_3(self):

        # set test parameters
        charges = np.ones(10).astype(np.float32)
        lj_sigmas = np.ones(10).astype(np.float32)
        lj_epsilons = np.ones(10).astype(np.float32)

        para = Parameters(temperature=1, box=np.array([1,1,1]), es_sigma=1, cutoff_radius=1, update_radius=1,
                          K_cutoff=1, charges=charges, lj_sigmas=lj_sigmas, lj_epsilons=lj_epsilons,
                          update_probability=0.5)

        particle1 = Particle(np.array([0.5, 0.5, 0.5]))
        particle2 = Particle(np.array([0.5, 0.5, 0.1]))
        particles = [particle1, particle2]

        system = System(particles=particles, parameters=para)
        system.energy.overall_energy = 1.1
        trial_system = System(particles=particles, parameters=para)
        trial_system.energy.overall_energy = 1

        actual = MetropolisMonteCarlo.evaluate_trial_configuration_greedy(system,trial_system)

        np.array_equal(actual, trial_system)
        
    def test_evaluate_trial_configuration_1(self):

        # set test parameters
        charges = np.ones(10).astype(np.float32)
        lj_sigmas = np.ones(10).astype(np.float32)
        lj_epsilons = np.ones(10).astype(np.float32)

        para = Parameters(temperature=1, box=np.array([1,1,1]), es_sigma=1, cutoff_radius=1, update_radius=1,
                          K_cutoff=1, charges=charges, lj_sigmas=lj_sigmas, lj_epsilons=lj_epsilons,
                          update_probability=0.5)

        particle1 = Particle(np.array([0.5, 0.5, 0.5]))
        particle2 = Particle(np.array([0.5, 0.5, 0.1]))
        particles = [particle1, particle2]

        system = System(particles=particles, parameters=para)
        system.energy.overall_energy = 1
        trial_system = System(particles=particles, parameters=para)
        trial_system.energy.overall_energy = 1

        actual = MetropolisMonteCarlo.evaluate_trial_configuration(system,trial_system, para)

        np.array_equal(actual, trial_system)

    def test_shift_position_1(self):

        position = np.array([0.5, 0.5, 0.5])

        # set test parameters
        charges = np.ones(10).astype(np.float32)
        lj_sigmas = np.ones(10).astype(np.float32)
        lj_epsilons = np.ones(10).astype(np.float32)

        para = Parameters(temperature=1, box=np.array([1,1,1]), es_sigma=1, cutoff_radius=1, update_radius=1,
                          K_cutoff=1, charges=charges, lj_sigmas=lj_sigmas, lj_epsilons=lj_epsilons,
                          update_probability=0.5)

        actual = MetropolisMonteCarlo._shift_position(position, para)

        np.array_equal(actual, position)

    def test_shift_position_2(self):

        position = np.array([1.5, 1.5, 1.5])

        # set test parameters
        charges = np.ones(10).astype(np.float32)
        lj_sigmas = np.ones(10).astype(np.float32)
        lj_epsilons = np.ones(10).astype(np.float32)

        para = Parameters(temperature=1, box=np.array([1,1,1]), es_sigma=1, cutoff_radius=1, update_radius=1,
                          K_cutoff=1, charges=charges, lj_sigmas=lj_sigmas, lj_epsilons=lj_epsilons,
                          update_probability=0.5)

        actual = MetropolisMonteCarlo._shift_position(position, para)
        reference = np.array([0.5, 0.5, 0.5])

        np.array_equal(actual, reference)

    def test_shift_position_3(self):

        position = np.array([-0.5, -0.5, -0.5])

        # set test parameters
        charges = np.ones(10).astype(np.float32)
        lj_sigmas = np.ones(10).astype(np.float32)
        lj_epsilons = np.ones(10).astype(np.float32)

        para = Parameters(temperature=1, box=np.array([1,1,1]), es_sigma=1, cutoff_radius=1, update_radius=1,
                          K_cutoff=1, charges=charges, lj_sigmas=lj_sigmas, lj_epsilons=lj_epsilons,
                          update_probability=0.5)

        actual = MetropolisMonteCarlo._shift_position(position, para)
        reference = np.array([0.5, 0.5, 0.5])

        np.array_equal(actual, reference)

    def test_shift_position_4(self):

        position = np.array([-1.3, 2.4, -3.4])

        # set test parameters
        charges = np.ones(10).astype(np.float32)
        lj_sigmas = np.ones(10).astype(np.float32)
        lj_epsilons = np.ones(10).astype(np.float32)

        para = Parameters(temperature=1, box=np.array([1,1,1]), es_sigma=1, cutoff_radius=1, update_radius=1,
                          K_cutoff=1, charges=charges, lj_sigmas=lj_sigmas, lj_epsilons=lj_epsilons,
                          update_probability=0.5)

        actual = MetropolisMonteCarlo._shift_position(position, para)
        reference = np.array([0.7, 0.4, 0.6])

        np.array_equal(actual, reference)

    # test if the specified update radius is actually satisfied
    def test_generate_trial_position_1(self):

        position = np.array([0.5, 0.5, 0.5])

        # set test parameters
        charges = np.ones(10).astype(np.float32)
        lj_sigmas = np.ones(10).astype(np.float32)
        lj_epsilons = np.ones(10).astype(np.float32)

        para = Parameters(temperature=1, box=np.array([1,1,1]), es_sigma=1, cutoff_radius=1, update_radius=1,
                          K_cutoff=1, charges=charges, lj_sigmas=lj_sigmas, lj_epsilons=lj_epsilons,
                          update_probability=0.5)

        for i in range(1000):
            trial_position = MetropolisMonteCarlo._generate_trial_position(position, para)
            distance = np.linalg.norm(position - trial_position)
            if distance > para.update_radius:
                raise AssertionError("Distance greater than update radius")

    # test if trial positions are uniformly distributed
    def test_generate_trial_position_2(self):

        position = np.array([0.5, 0.5, 0.5])

        # set test parameters
        charges = np.ones(10).astype(np.float32)
        lj_sigmas = np.ones(10).astype(np.float32)
        lj_epsilons = np.ones(10).astype(np.float32)

        para = Parameters(temperature=1, box=np.array([1, 1, 1]), es_sigma=1, cutoff_radius=1, update_radius=1,
                          K_cutoff=1, charges=charges, lj_sigmas=lj_sigmas, lj_epsilons=lj_epsilons,
                          update_probability=0.5)

        distances = []
        for i in range(10000):
            trial_position = MetropolisMonteCarlo._generate_trial_position(position, para)
            distances.append(np.linalg.norm(position - trial_position))

        distances = np.array(distances)
        npt.assert_approx_equal(distances.sum()/10000, 0.5, significant=2)