from Particle_Simulation.MetropolisMonteCarlo import MetropolisMonteCarlo
from Particle_Simulation.Parameters import Parameters
from Particle_Simulation.ParticleType import ParticleType
from Particle_Simulation.Particle import Particle
from Particle_Simulation.System import System
import numpy as np
import numpy.testing as npt
import unittest

class TestMetropolisMonteCarlo(unittest.TestCase):
    
    
    #check_same_position_input_behavior 
    def test__generate_trial_position_redundancy(self):
        particle_type = ParticleType(name="Natrium", mass=2, charge=2, lj_epsilon=1.25, lj_sigma=0.5)
        particle_type = np.array([particle_type])
        parameters = Parameters(temperature=0, box=np.array([12., 13., 14.]), es_sigma=0.5, update_radius=1,
                                 particle_types=particle_type, cutoff_radius=0.5)
        particle_1 = Particle(position=np.array([1, 2, 3]), type_index=0)
        particle_2 = Particle(position=np.array([1, 2, 3]), type_index=0)
        trial_position1 = np.array(MetropolisMonteCarlo._generate_trial_position(particle_1.position, parameters))
        trial_position2 = np.array(MetropolisMonteCarlo._generate_trial_position(particle_1.position, parameters))
        x = 0
        if  np.array_equal(trial_position1,trial_position2):
            x = 1
        else :
             x = 2
        npt.assert_equal(x, 2, 'Failed', verbose=True)
    
    # box and particle dimension mismatch
    @unittest.expectedFailure
    def test__generate_trial_position_2(self):
        particle_type = ParticleType(name="Hydrogen", mass=1.008, charge=1.602, lj_epsilon=0.21, lj_sigma= 2.5)
        particle_type = np.array([particle_type])
        parameters = Parameters(temperature=0, box=np.array([12., 13.]), es_sigma=0.5, update_radius=1,
                                 particle_types=particle_type, cutoff_radius=0.5)
        particle_1 = Particle(position=np.array([1, 2, 3]), type_index=0)
        
        trial_position1 = np.array(MetropolisMonteCarlo._generate_trial_position(particle_1.position, parameters))
    
    #type_index_negative
    @unittest.expectedFailure    
    def test__generate_trial_position_3(self):
        particle_type = ParticleType(name="Hydrogen", mass=1.008, charge=1.602, lj_epsilon=0.21, lj_sigma= 2.5)
        particle_type = np.array([particle_type])
        parameters = Parameters(temperature=0, box=np.array([12., 13., 13.]), es_sigma=0.5, update_radius=1,
                                 particle_types=particle_type, cutoff_radius=0.5)
        particle_1 = Particle(position=np.array([1, 2, 4]), type_index=1.5)
        trial_position1 = np.array(MetropolisMonteCarlo._generate_trial_position(particle_1.position, parameters))
        
    #negative_temprature
    def test__generate_trial_position_4(self):
        particle_type = ParticleType(name="Hydrogen", mass=1.008, charge=1.602, lj_epsilon=0.21, lj_sigma=2.5)
        particle_type = np.array([particle_type])
        parameters = Parameters(temperature=-1, box=np.array([12., 13., 13.]), es_sigma=0.5, update_radius=1,
                                 particle_types=particle_type, cutoff_radius=0.5)
        particle_1 = Particle(position=np.array([1, 2, 4]), type_index=1.5)
        trial_position1 = np.array(MetropolisMonteCarlo._generate_trial_position(particle_1.position, parameters))
            
    #negaive_vlaue_sigma
    @unittest.expectedFailure
    def test__generate_trial_position_5(self):
        particle_type = ParticleType(name="test_element", mass=1.008, charge=1.602, lj_epsilon=0.21, lj_sigma= -2.5)
        particle_type = np.array([particle_type])
        parameters = Parameters(temperature=120, box=np.array([12., 13., 13.]), es_sigma= -0.5, update_radius=1,
                                 particle_types=particle_type, cutoff_radius=0.5)
        particle_1 = Particle(position=np.array([1, 2, 4]), type_index=1.5)
        trial_position1 = np.array(MetropolisMonteCarlo._generate_trial_position(particle_1.position, parameters))
        
    #negaive_value_cutoff
    @unittest.expectedFailure
    def test__generate_trial_position_6(self):
        particle_type = ParticleType(name="Hydrogen", mass=1.008, charge=1.602, lj_epsilon=0.21, lj_sigma= 2.5)
        particle_type = np.array([particle_type])
        parameters = Parameters(temperature=120, box=np.array([12., 13., 13.]), es_sigma= -0.5, update_radius=1,
                                 particle_types=particle_type, cutoff_radius= -0.5)
        particle_1 = Particle(position=np.array([1, 2, 4]), type_index=1.5)
        trial_position1 = np.array(MetropolisMonteCarlo._generate_trial_position(particle_1.position, parameters))
        
    def test_shift_position(self):
        particle_type = ParticleType(name="Hydrogen", mass=1.008, charge=1.602, lj_epsilon=0.21, lj_sigma= 2.5)
        particle_type = np.array([particle_type])
        
        parameters = Parameters(temperature=0, box=np.array([12., 13., 14.]), es_sigma=0.5, update_radius=1,
                                 particle_types=particle_type, cutoff_radius=3)
        reference_pos1 =np.array([ 1.5,  1.0,   1.5])
        reference_pos2 =np.array([ 10.5,  12.0,   5.5])
        particle_position_1=np.array([13.5, 14., 15.5])
        shifted_position_1 = MetropolisMonteCarlo._shift_position(particle_position_1,parameters)
        particle_position_2=np.array([-1.5, -1., 5.5])
        shifted_position_2 = MetropolisMonteCarlo._shift_position(particle_position_2,parameters)
        npt.assert_equal(reference_pos1, shifted_position_1, 'Failed', verbose=True)
        npt.assert_equal(reference_pos2, shifted_position_2, 'Failed', verbose=True)
    
    #to check behavior of trail_positions 
#     def test_generate_trial_position_behavior(self):
#         particle_type = ParticleType(name="Natrium", mass=2, charge=2, lj_epsilon=1.25, lj_sigma=0.5)
#         particle_type = np.array([particle_type])
#         parameters = Parameters(temperature=0, box=np.array([12., 13., 14.]), es_sigma=0.5, update_radius=1,
#                                 particle_types=particle_type, cutoff_radius=0.5)
#         particle_1 = Particle(position=np.array([0 ,0, 0]), type_index=0)
#         trial_position1 = np.zeros(3)
#         for i in range(100):
#             trial_position1 = np.array(MetropolisMonteCarlo._generate_trial_position(particle_1.position, parameters))
#             print(trial_position1)
