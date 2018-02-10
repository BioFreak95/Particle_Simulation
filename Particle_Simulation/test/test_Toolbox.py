import numpy as np
import unittest
import numpy.testing as npt
from Particle_Simulation.ToolBox import ToolBox

class test_ToolBox(unittest.TestCase):
    
    # parameter fetch test
    def test_get_inputs(self):
        parameters_input = {'A' : [3.32, 0.011589, 22.98, 1.6],
                      'B' : [4.40104, 0.4184, 35.453, 2.3]}
        ref_types =  ['A', 'B', 'B', 'A']
        ref_position = np.asarray([[1.5, 1., 2.5],
                                [1.5, 4., 3.5],
                                [1.5, 4., 3.],
                                [1.5, 4., 4.]])
        ref_box = np.asarray([20.0, 20.0, 20.0])
        #save file
        np.savez('Test_Inputs.npz', parameters=parameters_input, types=ref_types, positions=ref_position, box=ref_box)
        #work with DirectoryPath also
        file_path = 'Test_Inputs.npz'
        ref_name = np.array(['A', 'B'])
        ref_lj_sigmas = np.array([3.32, 4.40104 ])
        ref_lj_epsilons = np.array([0.011589, 0.4184 ])
        ref_charges = np.array([1.6, 2.3 ])
        ref_mass = np.array([22.98, 35.453 ])
        particle, particle_positions, box, lj_sigmas, lj_epsilons, charges, name, mass = ToolBox.get_inputs(file_path)
        npt.assert_equal(particle[0].position, ref_position[0], 'Failed', verbose=True)
        npt.assert_equal(ref_position, particle_positions, 'Failed', verbose=True)
        npt.assert_equal(ref_position, particle_positions, 'Failed', verbose=True)
        npt.assert_equal(ref_box, box, 'Failed', verbose=True)
        npt.assert_equal(ref_lj_sigmas, lj_sigmas, 'Failed', verbose=True)
        npt.assert_equal(ref_lj_epsilons, lj_epsilons, 'Failed', verbose=True)
        npt.assert_equal(ref_charges, charges, 'Failed', verbose=True)
        npt.assert_equal(ref_name, name, 'Failed', verbose=True)
        npt.assert_equal(ref_mass, mass, 'Failed', verbose=True)