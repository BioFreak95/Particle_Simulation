import csv
import numpy as np
from numpy import float64
from Particle_Simulation.Particle import Particle
from Particle_Simulation.ParticleType import ParticleType
from Particle_Simulation.Parameters import Parameters

class Toolbox:
    
    #Read data from Input_Particle.csv
    def get_particle(particle_file_path):
        input_particle = np.genfromtxt(particle_file_path, delimiter=",", skip_header=1, autostrip=True)
        type_index = (input_particle[: , 0]).astype(int)
        position = input_particle[ : , 1 : ] 
        #Particle Object Creation
        particle_list = []
        for i in range(len(input_particle)):
            particle_obj = Particle(type_index=type_index[i], position=position[i])
            particle_list.append(particle_obj)
        particle = np.array(particle_list)
        return particle
    
    #Read data from Input_PaticleType.csv
    def get_particle_type(particle_typ_file_path):
        in_particle_type = np.genfromtxt(particle_typ_file_path, delimiter=",", skip_header=1, autostrip=True, dtype='U')
        name = in_particle_type[ : , 0]
        mass = (in_particle_type[ : , 1]).astype(np.float64) 
        charge = (in_particle_type[ : ,2]).astype(np.float64)
        lj_epsilon = (in_particle_type[ : , 3]).astype(np.float64)
        lj_sigma = (in_particle_type[ : , 4]).astype(np.float64)
        particle_type_list = []
        for i in range(len(in_particle_type)):
            particle_type_obj = ParticleType(name=name[i], mass=mass[i], charge=charge[i],
                                             lj_epsilon=lj_epsilon[i], lj_sigma=lj_sigma[i])
            particle_type_list.append(particle_type_obj)
        particle_type = np.array(particle_type_list)
        return particle_type
    
    #Read data from Input_Parameter.csv
    def get_parameters(parameters_file_path, particle_typ_file_path):
        in_parameters = np.genfromtxt(parameters_file_path, delimiter=",", skip_header=1, autostrip=True)
        temperature = in_parameters[ : , 0]
        box = in_parameters[ : , 1 : 4 ]
        es_sigma = in_parameters[ : , 4]
        update_radius = in_parameters[ : , 5]
        cutoff_radius = in_parameters[ : , 6]
        parameters_list = []
        particle_types=Toolbox.get_particle_type(particle_typ_file_path)
        for i in range(len(in_parameters)):
            parameters__obj = Parameters(temperature=temperature[i], box=box[i], es_sigma=es_sigma[i], 
                                         update_radius=update_radius[i], particle_types=particle_types, cutoff_radius=cutoff_radius[i])
            parameters_list.append(parameters__obj)
        parameters = np.array(parameters_list)
        return parameters
    
    if __name__ == '__main__':
        print('Toolbox running')