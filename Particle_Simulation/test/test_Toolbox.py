import numpy as np
import unittest
import numpy.testing as npt
from Particle_Simulation.ToolBox import ToolBox


class test_ToolBox(unittest.TestCase):
    # parameter fetch test
    def test_get_inputs(self):
        parameters_input = {'Na+': [3.3284, 0.0115897, 22.9898, 1.0],
                            'Cl-': [4.40104, 0.4184, 35.453, -1.0]}
        ref_position = np.asarray([[1.5, 1., 2.5],
                                   [1.5, 4., 3.5],
                                   [1.5, 4., 3.],
                                   [1.5, 4., 4.]])
        ref_types = ['Na+', 'Cl-', 'Cl-', 'Na+']
        ref_box = np.asarray([20.0, 20.0, 20.0])
        readme = 'values specifications'
        # save file
        np.savez('Test_Inputs.npz', parameters=parameters_input, positions=ref_position, types=ref_types, box=ref_box,
                 readme=readme)
        # work with DirectoryPath also
        file_path = 'Test_Inputs.npz'
        ref_name = np.array(['Na+', 'Cl-', 'Cl-', 'Na+'])
        ref_lj_sigmas = np.array([3.3284, 4.40104, 4.40104, 3.3284])
        ref_lj_epsilons = np.array([0.0115897, 0.4184, 0.4184, 0.0115897])
        ref_charges = np.array([1, -1, -1, 1])
        ref_mass = np.array([22.9898, 35.453, 35.453, 22.9898])
        particle, box, particle_positions, types, name, lj_sigmas, lj_epsilons, mass, charges, readme = ToolBox.get_inputs(
            file_path)

        # assert statements
        npt.assert_equal(particle[0].position, ref_position[0], 'Failed', verbose=True)
        npt.assert_equal(ref_position, particle_positions, 'Failed', verbose=True)
        npt.assert_equal(ref_position, particle_positions, 'Failed', verbose=True)
        npt.assert_equal(ref_types, types, 'Failed', verbose=True)
        npt.assert_equal(ref_box, box, 'Failed', verbose=True)
        npt.assert_equal(ref_lj_sigmas, lj_sigmas, 'Failed', verbose=True)
        npt.assert_equal(ref_lj_epsilons, lj_epsilons, 'Failed', verbose=True)
        npt.assert_equal(ref_charges, charges, 'Failed', verbose=True)
        npt.assert_equal(ref_name, name, 'Failed', verbose=True)
        npt.assert_equal(ref_mass, mass, 'Failed', verbose=True)
