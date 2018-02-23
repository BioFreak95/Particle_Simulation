import numpy as np
import imageio
from Particle_Simulation.ToolBox import ToolBox
from Particle_Simulation.Parameters import Parameters
from Particle_Simulation.System import System
from Particle_Simulation.Simulation import Simulation
from Particle_Simulation.Particle import Particle

# read data
data = np.load('/home/max/Downloads/sodium-chloride-example.npz')
print(data['parameters'])

particles, box, particle_positions, types, name, lj_sigmas, lj_epsilons, mass, charges, readme = ToolBox.get_inputs(
    '/home/max/Downloads/sodium-chloride-example.npz')

avogadro_constant = 6.02214085774 * 10 ** (23)

# scale parameters
lj_epsilons = lj_epsilons * 1000 / avogadro_constant
lj_sigmas = lj_sigmas / 10
particle_positions = particle_positions / 10
box = box / 10
particles = []

# create particle array
for i in range(len(particle_positions)):
    particle_obj = Particle(position=particle_positions[i])
    particles.append(particle_obj)

particles = np.array(particles)

# initialize parameters object
parameters = Parameters(temperature=200, box=box, charges=charges, lj_epsilons=lj_epsilons, lj_sigmas=lj_sigmas,
                        accuracy=2)


print("Starting simulation with the following parameters:")
print("Temperature: ", parameters.temperature)
print("Accuracy: ", parameters.accuracy)
print("Cutoff radius: ", parameters.cutoff_radius)
print("K cutoff: ", parameters.K_cutoff)
print("Update probability: ", parameters.update_probability)
print("Update radius: ", parameters.update_radius)
print("ES sigma: ", parameters.es_sigma)

initial_system = System(particles, parameters)
# ToolBox.plot_system(initial_system, parameters)

steps_opt = 5000
steps_sim = 2000

simulation = Simulation(initial_system, parameters)
simulation.parameters.update_radius = np.sum(simulation.parameters.box) / len(simulation.parameters.box) * 0.0005
simulation.optimize(steps_opt)
simulation.parameters.temperature = 1000
simulation.parameters.update_radius = np.sum(simulation.parameters.box) / len(simulation.parameters.box) * 0.0005
simulation.simulate(steps_sim)

i = 0
j = 0
ToolBox.plot_overall_energy_trend(simulation.opt_systems)
ToolBox.plot_overall_energy_trend(simulation.sim_systems)

ToolBox.plot_system(simulation.opt_systems[-1], simulation.parameters)

while i < steps_sim:
    ToolBox.save_system(simulation.sim_systems[i], simulation.parameters, j)
    i += 5
    j += 1

images2 = []
for i in range(j):
    images2.append('tmp/sys' + str(i) + '.png')

with imageio.get_writer('movie.gif', mode='I', duration=0.05) as writer:
    for filename in images2:
        image = imageio.imread(filename)
        writer.append_data(image)

'''
initial_system = System(particles, parameters)
steps_opt = 1000
simulation = Simulation(initial_system, parameters)
simulation.optimize(steps_opt)
ToolBox.calculate_rdf(simulation.opt_systems[-1], parameters, 0.001, 0.01)

'''
