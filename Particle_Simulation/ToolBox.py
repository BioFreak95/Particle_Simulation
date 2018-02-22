import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np

from Particle import Particle
from System import System
from Simulation import Simulation
from Parameters import Parameters

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
    def plot_system(system, parameters):

        x = []
        y = []
        z = []
        colours = []
        for i in range(len(system.particles)):
            x.append(system.particles[i].position[0])
            y.append(system.particles[i].position[1])
            z.append(system.particles[i].position[2])

            if parameters.charges[i] == 1:
                colours.append('b')
            elif parameters.charges[i] == -1:
                colours.append('g')

        ax = plt.axes(projection='3d')
        ax.set_xlim3d(0, system.neighbourlist.box_space[0])
        ax.set_ylim3d(0, system.neighbourlist.box_space[1])
        ax.set_zlim3d(0, system.neighbourlist.box_space[2])
        Axes3D.scatter(ax, x, y, z, c=colours)
        plt.title("System")
        #ax.legend(['Sodium','Chloride'], loc='upper right')

        plt.show()

    @staticmethod
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

    @staticmethod
    def setup_random_simulation(box_length):

        cr = []
        for i in range(box_length):
            cr.append(i + 0.5)

        x, y, z = np.meshgrid(cr, cr, cr)
        xyz = np.vstack((x.flat, y.flat, z.flat))
        xyz = np.ascontiguousarray(xyz)

        particles = []
        for i in range(len(xyz[0, :])):
            particles.append(Particle(np.array(xyz[:, i])))

        charges = np.zeros(len(particles))
        lj_sigmas = np.zeros(len(particles))
        lj_epsilons = np.zeros(len(particles))

        avogadro_constant = 6.02214085774 * 10 ** (23)
        for i in range(len(particles)):
            if i % 2 == 0:
                charges[i] = 1
                lj_sigmas[i] = 0.33
                lj_epsilons[i] = 11.6 / avogadro_constant
            elif i % 2 == 1:
                charges[i] = -1
                lj_sigmas[i] = 0.44
                lj_epsilons[i] = 418.4 / avogadro_constant

        parameters = Parameters(temperature=300, box=np.array([box_length, box_length, box_length]), es_sigma=0.223,
                                charges=charges, lj_sigmas=lj_sigmas, lj_epsilons=lj_epsilons, accuracy=10)

        system = System(particles, parameters)
        simulation = Simulation(system, parameters)

        return simulation