import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np

from Particle_Simulation.Particle import Particle
from Particle_Simulation.System import System
from Particle_Simulation.Simulation import Simulation
from Particle_Simulation.Parameters import Parameters


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
        # ax.legend(['Sodium','Chloride'], loc='upper right')

        plt.show()

    @staticmethod
    def get_inputs(file_path):
        # Load arrays from .npz file
        with np.load(file_path) as fh:
            box = fh['box']
            particle_positions = fh['positions']
            types = fh['types']
            readme = fh['readme']
            # scalarization (parameters.npy to dictionary)
            parameters = fh['parameters'].item()
            # fetch values from dictionary and create input ndarrays
            prmtr_vals = []
            for i in range(len(types)):
                prmtr_vals.append(parameters.get(types[i]))
            prmtr_vals = np.asarray(prmtr_vals)

            name = types
            lj_sigmas = prmtr_vals[:, 0]
            lj_epsilons = prmtr_vals[:, 1]
            mass = prmtr_vals[:, 2]
            charges = prmtr_vals[:, 3]
            # Particle object:
            particle_list = []
            for i in range(len(particle_positions)):
                particle_obj = Particle(position=particle_positions[i])
                particle_list.append(particle_obj)
            particle = np.array(particle_list)

            return particle, box, particle_positions, types, name, lj_sigmas, lj_epsilons, mass, charges, readme

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

        parameters = Parameters(temperature=300, box=np.array([box_length, box_length, box_length]),
                                charges=charges, lj_sigmas=lj_sigmas, lj_epsilons=lj_epsilons, accuracy=10)

        system = System(particles, parameters)
        simulation = Simulation(system, parameters)

        return simulation

    @staticmethod
    def save_system(system, parameters, num):

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
        # ax.legend(['Sodium','Chloride'], loc='upper right')

        plt.savefig('tmp/sys' + str(num) + '.png')

    @staticmethod
    def _wrap_distance(distance, box):
        for i in range(len(distance)):
            while distance[i] >= 0.5 * box[i]:
                distance[i] -= box[i]
            while distance[i] < -0.5 * box[i]:
                distance[i] += box[i]

        return distance

    @staticmethod
    def calculate_rdf(system, parameters, steprange, thickness):
        # rang = int(np.floor((np.sum(parameters.box) / len(parameters.box)) / steprange))
        rang = 3000

        distances = np.zeros(rang)
        particle_neighbors = np.zeros(rang)

        particle = system.particles[0]
        shell_thickness = thickness

        for i in range(rang):

            distance = steprange * i
            for j in range(1, len(system.particles)):
                particle_distance = np.linalg.norm(
                    ToolBox._wrap_distance(np.asarray(particle.position) - np.asarray(system.particles[j].position),
                                           parameters.box))
                if particle_distance > distance and particle_distance < distance + shell_thickness and \
                                parameters.charges[0] != parameters.charges[j]:
                    particle_neighbors[i] += 1

            particle_neighbors[i] = particle_neighbors[i] / (
                4 / 3 * np.pi * ((distance + shell_thickness) ** 3 - (distance) ** 3))
            distances[i] = distance

        rdf = particle_neighbors / (len(system.particles) / np.prod(parameters.box))

        plt.plot(distances, rdf)
        plt.xlabel("Distance")
        plt.ylabel("RDF")
        plt.show()
