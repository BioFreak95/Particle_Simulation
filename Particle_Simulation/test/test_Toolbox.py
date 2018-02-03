import numpy as np
import unittest
import numpy.testing as npt
from Particle_Simulation.Toolbox import Toolbox


class test_ToolBox(unittest.TestCase):
    # parameter fetch test    
    def test_print_inputs(self):
        parameters = {'A' : [3.32, 0.011589, 22.98, 1.6],
                      'B' : [4.40104, 0.4184, 35.453, 2.3]}
        types =  ['A', 'B', 'B', 'A']
        positions = np.asarray([[1.5, 1., 2.5],
                                [1.5, 4., 2.5],
                                [1.5, 4., 3.],
                                [1.5, 4., 4.]])
        box = np.asarray([20.0, 20.0, 20.0])
        temprature = 120
        update_radius = 2.
        es_sigma = 3.5
        k_cutoff = 4.
        parameters, types, positions, box = Toolbox.get_inputs(parameters, types, positions, box)
        ref_parameters = {'A': [3.32, 0.011589, 22.98, 1.6], 'B': [4.40104, 0.4184, 35.453, 2.3]}
        ref_types = np.array(types)
        ref_positions = np.array(positions)
        ref_box = np.asarray(box)
        self.assertEqual(ref_parameters, parameters, msg='Failed: Actual and Desired are not equal')
        npt.assert_equal(ref_types, types, 'Failed', verbose=True)
        npt.assert_equal(ref_positions, positions, 'Failed', verbose=True)
        npt.assert_equal(ref_box, box, 'Failed', verbose=True)

    def test_get_particle(self):
        parameters = {'A' : [3.32, 0.011589, 22.98, 1.6],
                      'B' : [4.40104, 0.4184, 35.453, 2.3]}
        types =  ['A', 'B', 'B', 'A']
        positions = np.asarray([[1.5, 1., 2.5],
                                [1.5, 4., 2.5],
                                [1.5, 4., 3.],
                                [1.5, 4., 4.]])
        box = np.asarray([20.0, 20.0, 20.0])
        temprature = 120
        update_radius = 2.
        es_sigma = 3.5
        k_cutoff = 4.
        type_index_cal, position_cal, particle = Toolbox.get_particle(parameters, types, positions, box)
        ref_positions = np.array(positions)
        ref_type_index = np.array([0, 1, 1, 0])
        ref_position = np.array(positions)
        npt.assert_equal(ref_type_index, type_index_cal, 'Failed', verbose=True)
        npt.assert_equal(ref_position, position_cal, 'Failed', verbose=True)
    
    def test_get_particle_type(self):
        parameters = {'A' : [3.32, 0.011589, 22.98, 1.6],
                      'B' : [4.40104, 0.4184, 35.453, 2.3]}
        types =  ['A', 'B', 'B', 'A']
        positions = np.asarray([[1.5, 1., 2.5],
                                [1.5, 4., 2.5],
                                [1.5, 4., 3.],
                                [1.5, 4., 4.]])
        box = np.asarray([20.0, 20.0, 20.0])
        temprature = 120
        update_radius = 2.
        es_sigma = 3.5
        k_cutoff = 4.
        particle_type, name, mass, charge, lj_epsilon, lj_sigma  = Toolbox.get_particle_type(parameters, 
                                                                                             types, positions, box)
        ref_name = np.array(['A', 'B'])
        ref_lj_sigma = np.array([3.32, 4.40104 ])
        ref_lj_epsilon = np.array([0.011589, 0.4184 ])
        ref_mass = np.array([22.98, 35.453 ])
        ref_charge = np.array([1.6, 2.3 ])
        npt.assert_equal(ref_name, name, 'Failed', verbose=True)
        npt.assert_equal(ref_lj_sigma, lj_sigma, 'Failed', verbose=True)
        npt.assert_equal(ref_lj_epsilon, lj_epsilon, 'Failed', verbose=True)
        npt.assert_equal(ref_mass, mass, 'Failed', verbose=True)
        npt.assert_equal(ref_charge, charge, 'Failed', verbose=True)
        #print(particle_type[0].mass)
        
    def test_get_parameters(self):
        parameters = {'A' : [3.32, 0.011589, 22.98, 1.6],
                      'B' : [4.40104, 0.4184, 35.453, 2.3]}
        types =  ['A', 'B', 'B', 'A']
        positions = np.asarray([[1.5, 1., 2.5],
                                [1.5, 4., 2.5],
                                [1.5, 4., 3.],
                                [1.5, 4., 4.]])
        box = np.asarray([20.0, 20.0, 20.0])
        temprature = 120
        update_radius = .5
        cutoff_radius = 3.
        es_sigma = 3.5
        k_cutoff = 2
        parameters_obj = Toolbox.get_parameters(parameters, types, positions, box, 
                                                temprature, update_radius,cutoff_radius, es_sigma, k_cutoff)
        ref_temperature = 120
        ref_box = np.asarray(box)
        npt.assert_equal(ref_box, parameters_obj.box, 'Failed', verbose=True)
        npt.assert_equal(ref_temperature,parameters_obj.temperature, 'Failed', verbose=True)
        
    if __name__ == '__main__':
        unittest.main()
    