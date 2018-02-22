import numpy as np
import imageio
from Particle_Simulation.ToolBox import ToolBox
from Particle_Simulation.Parameters import Parameters
from Particle_Simulation.System import System
from Particle_Simulation.Simulation import Simulation

data = np.load('/home/max/Downloads/sodium-chloride-example.npz')
print(data['parameters'])

particles, box, particle_positions, types, name, lj_sigmas, lj_epsilons, mass, charges, readme = ToolBox.get_inputs(
    '/home/max/Downloads/sodium-chloride-example.npz')

avogadro_constant = 6.02214085774 * 10 ** (23)
lj_epsilons = lj_epsilons * 1000 / avogadro_constant
lj_sigmas = lj_sigmas / 10

parameters = Parameters(temperature=300, box=box, charges=charges, lj_epsilons=lj_epsilons, lj_sigmas=lj_sigmas,
                        accuracy=10)

print("Starting simulation with the following parameters:")
print("Temperature: ", parameters.temperature)
print("Accuracy: ", parameters.accuracy)
print("Cutoff radius: ", parameters.cutoff_radius)
print("K cutoff: ", parameters.K_cutoff)
print("Update probability: ", parameters.update_probability)
print("Update radius: ", parameters.update_radius)
print("ES sigma: ", parameters.es_sigma)

initial_system = System(particles, parameters)

steps_opt = 7500
steps_sim = 2500

simulation = Simulation(initial_system, parameters)
simulation.optimize(steps_opt)
simulation.simulate(steps_sim)

i = 0
j = 0
ToolBox.plot_overall_energy_trend(simulation.opt_systems)
ToolBox.plot_overall_energy_trend(simulation.sim_systems)

ToolBox.plot_system(simulation.opt_systems[-1], simulation.parameters)

while i < steps_sim:
    ToolBox.save_system(simulation.sim_systems[i], simulation.parameters, j)
    i += 1
    j += 1

images2 = []
for i in range(j):
    images2.append('tmp/sys' + str(i) + '.png')
    '''
images = []
for filename in images2:
    images.append(imageio.imread(filename))
imageio.mimsave('movie.gif', images, duration=0.01)
'''
with imageio.get_writer('movie.gif', mode='I', duration=0.005) as writer:
    for filename in images2:
        image = imageio.imread(filename)
        writer.append_data(image)
