import os
import numpy as np
from numpy import float64, dtype
from Particle_Simulation.Particle import Particle
from Particle_Simulation.ParticleType import ParticleType
from Particle_Simulation.Parameters import Parameters

# To_do to handle charge.temperature,update_radius,cut_off radius
# taken charge as fourth input for parameters
class Toolbox:
    
    def get_inputs(parameters, types, positions, box):
        #Saving input arrays into a single file in uncompressed .npz format
        np.savez('Inputs.npz', parameters=parameters, types=types, positions=positions, box=box)
        #Loading arrays from .npz format
        with np.load('Inputs.npz') as fh :
            parameters_fh = fh['parameters'] 
            types_fh = fh['types']
            positions_fh = fh['positions']
            box_fh = fh['box']
            return parameters_fh, types_fh, positions_fh, box_fh
    
    #Particle class variables creation:
    def get_particle(parameters, types, positions, box):
        #Load input arrays from .npz format
        parameters_fh, types_fh, positions_fh, box_fh = Toolbox.get_inputs(parameters, types, positions, box)
        #particle_name to integer value mapping for type_index 
        #dictionary creation like {'A': 0, 'B': 1}
        particle_names = list(parameters.keys()) # key part of parameters
        dict_name_to_type= {}
        for i, type_name  in enumerate(particle_names):
            dict_name_to_type[type_name] = i
        #type_index Integer array creation
        type_index = [dict_name_to_type[name_key] for name_key in types_fh]
        type_index = np.array(type_index)
        position = positions_fh
        #particle object creation
        particle_list = []
        for i in range(len(type_index)):
            particle_obj = Particle(type_index=type_index[i], position=position[i])
            particle_list.append(particle_obj)
        particle = np.array(particle_list)
        os.remove('Inputs.npz')
        return type_index, position, particle
        
    #ParticleType class variables creation
    def get_particle_type(parameters, types, positions, box):
        parameters_fh, types_fh, positions_fh, box_fh = Toolbox.get_inputs(parameters, types, positions, box)
        # key part of parameters dictionary
        particle_names = list(parameters.keys()) 
        #value part of parameters.npy file made
        x = parameters_fh.item() 
        values = [x[name_key] for name_key in particle_names] 
        values = np.array(values)
        #ParticleType object creation
        name = np.array(particle_names)
        lj_sigma = values[ : , 0]
        lj_epsilon = values[ : , 1]
        mass = values[ : , 2]
        charge = values[ : , 3]
        particle_type_list = []
        for i in range(len(name)):
            particle_type_obj = ParticleType(name=name[i], mass=mass[i], charge=charge[i],
                                             lj_epsilon=lj_epsilon[i], lj_sigma=lj_sigma[i])
            particle_type_list.append(particle_type_obj)
        particle_type = np.array(particle_type_list)
        os.remove('Inputs.npz')
        return particle_type, name, mass, charge, lj_epsilon, lj_sigma  
    
    #Parameter class variables creation
    def get_parameters(parameters, types, positions, box, temprature, update_radius,cutoff_radius, es_sigma, k_cutoff):
        #Load input arrays from .npz format
        parameters_fh, types, positions, box = Toolbox.get_inputs(parameters, types, positions, box)
        #parameters object creation
        parameters_1 =parameters
        temprature_1 = temprature
        update_radius_1 = update_radius
        es_sigma_1 = es_sigma
        K_cutoff_1 = k_cutoff
        cutoff_radius_1 = cutoff_radius
        #get particle_types 
        particle_types, name, mass, charge, lj_epsilon, lj_sigma = Toolbox.get_particle_type(parameters_1, types, positions, box)
        parameters_obj = Parameters(temprature_1, box, es_sigma_1,
                                    update_radius_1, particle_types, cutoff_radius_1, K_cutoff_1)
        return parameters_obj
    
    if __name__ == '__main__':
        print('Toolbox running')

    
    