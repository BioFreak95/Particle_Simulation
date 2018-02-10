import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
from Particle_Simulation.Particle import Particle

class ToolBox:

    @staticmethod
    def plot_overall_energy_trend(trajectory):

        energies = []
        for i in range(len(trajectory)):
            energies.append(trajectory[i].energy.overall_energy)

        plt.plot(energies)
        plt.title("Overall Energy Trend")
        plt.xlabel("Step")
        plt.ylabel("Energy")
        plt.show()

    @staticmethod
    def plot_es_energy_contribution_trend(trajectory):

        es_shortranged_energies = []
        es_longranged_energies = []
        es_selfinteraction_energies = []

        for i in range(len(trajectory)):
            es_shortranged_energies.append(trajectory[i].energy.es_shortranged_energy)
        for i in range(len(trajectory)):
            es_longranged_energies.append(trajectory[i].energy.es_longranged_energy)
        for i in range(len(trajectory)):
            es_selfinteraction_energies.append(trajectory[i].energy.es_selfinteraction_energy)

        plt.plot(es_shortranged_energies)
        plt.plot(es_longranged_energies)
        plt.plot(es_selfinteraction_energies)

        plt.title("Ewald Summation Energy Contributions")
        plt.xlabel("Step")
        plt.ylabel("Energy")

        plt.legend(['ES Shortranged', 'ES Longranged', 'ES Selfinteraction'], loc='upper right')

        plt.show()

    @staticmethod
    def plot_energy_contribution_trend(trajectory):

        lj_energies = []
        es_energies = []
        for i in range(len(trajectory)):
            lj_energies.append(trajectory[i].energy.lj_energy)
        for i in range(len(trajectory)):
            es_energies.append(trajectory[i].energy.es_energy)

        plt.plot(lj_energies)
        plt.plot(es_energies)
        plt.title("Energy Contributions")
        plt.xlabel("Step")
        plt.ylabel("Energy")

        plt.legend(['Lennard-Jones', 'Ewald Summation'], loc='upper right')

        plt.show()

    @staticmethod
    def plot_system(system):

        x = []
        y = []
        z = []
        for i in range(len(system.particles)):
            x.append(system.particles[i].position[0])
            y.append(system.particles[i].position[1])
            z.append(system.particles[i].position[2])

        ax = plt.axes(projection='3d')
        Axes3D.scatter(ax, x, y, z)
        plt.show()
        
    def get_inputs(file_path):
        #Load arrays from .npz file
        with np.load(file_path) as fh :
            parameters = fh['parameters']
            types = fh['types']
            particle_positions = fh['positions']
            box = fh['box']
            #scalar (nd-array parameters.npy to dictionary)
            prmtr_dict = parameters.item()
            name =[]
            prmtr_vals =[]
            for k, v in prmtr_dict.items():
                name.append(k)
                prmtr_vals.append(v)
            name = np.array(name)
            prmtr_vals = np.array(prmtr_vals)
            lj_sigmas = prmtr_vals[ : , 0]
            lj_epsilons = prmtr_vals[ : , 1]
            mass = prmtr_vals[ : , 2]
            charges = prmtr_vals[ : , 3]
            #Particle object:
            particle_list = []
            for i in range(len(particle_positions)):
                particle_obj = Particle(position=particle_positions[i])
                particle_list.append(particle_obj)
            particle = np.array(particle_list)
            return particle, particle_positions, box, lj_sigmas, lj_epsilons, charges, name, mass