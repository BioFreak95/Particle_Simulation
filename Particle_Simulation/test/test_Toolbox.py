import os
import numpy as np
import unittest
from Particle_Simulation.LennardJones import LennardJones
from Particle_Simulation.Toolbox import Toolbox


class test_ToolBox(unittest.TestCase):
    
    #If InputFile in value of os.path.abspath or BuildDirectory
    def test_get_particle(self):
        reference_distance = 3.5
        particle = Toolbox.get_particle(os.path.abspath("Input_Particle.csv"))
        distance = LennardJones._calculate_distance(particle[0], particle[1])
        self.assertEqual(reference_distance, distance, msg='Failed: Actual and Desired are not equal')
        
    def test_get_parameters(self):
        reference_potential = -0.000042499
        particle = Toolbox.get_particle(os.path.abspath("Input_Particle.csv"))
        parameters = Toolbox.get_parameters("Input_Parameter.csv", "Input_ParticleType.csv")
        lg_sigma = LennardJones._determine_sigma(particle[0], particle[1], parameters[0])
        lg_value = LennardJones._calculate_potential(particle[0], particle[1], parameters[0])
        lg_value_rounded = lg_value.round(decimals=9)
        self.assertEqual(reference_potential, lg_value_rounded, msg='Failed: Actual and Desired are not equal')
         
    #If user inputs File path: 
    #Build failing bcoz of no user input so commented
    #However running 
#     def test_user_input_get_particle(self):
#         reference_distance = 3.5
#         particle_file_path = input("Enter the Particle File Path: ")
#         particle = Toolbox.get_particle(particle_file_path)
#         distance = LennardJones._calculate_distance(particle[0], particle[1])
#         self.assertEqual(reference_distance, distance, msg='Failed: Actual and Desired are not equal')
             
    if __name__ == '__main__':
        unittest.main()
        
        