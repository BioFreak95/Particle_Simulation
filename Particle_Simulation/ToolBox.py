import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

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