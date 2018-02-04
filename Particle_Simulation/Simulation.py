from Particle_Simulation.System import System
from Particle_Simulation.Energy import Energy
from Particle_Simulation.EnergyCalculator import EnergyCalculator
from Particle_Simulation.MetropolisMonteCarlo import MetropolisMonteCarlo


class Simulation:

    # - - - constructor - - - #

    def __init__(self, system, parameters):

        self.system = system
        self.parameters = parameters
        self.opt_systems = []
        self.sim_systems = []

        self.energy_calculator = EnergyCalculator(cutoff_radius=parameters.cutoff_radius,
                                                  es_sigma=parameters.es_sigma,
                                                  box=parameters.box,
                                                  charges=parameters.charges,
                                                  lj_sigmas=parameters.lj_sigmas,
                                                  lj_epsilons=parameters.lj_epsilons,
                                                  k_vector=parameters.k_vector)

        self.energy_calculator.cell_neighbour_list = System.cell_neighbour_list

    # - - - public methods - - - #

    def optimize(self, n_steps):

        # set up initial system
        current_system = System(self.system.particles, self.parameters)
        # pass particle positions and neighbourlist to the energy calculator class
        self.energy_calculator.set_system(current_system.neighbourlist.particle_positions,
                                          current_system.neighbourlist.cell_list,
                                          current_system.neighbourlist.particle_neighbour_list)
        # calculate energy of initial system
        current_system.energy = self._calculate_overall_energy()
        # append to optimize trajectory
        self.opt_systems.append(current_system)

        # crude optimization
        for i in range(n_steps):
            # generate trial system
            trial_system = MetropolisMonteCarlo.generate_trial_configuration(self.opt_systems[-1], self.parameters)

            # update particle positions and neighbourlist
            self.energy_calculator.set_system(trial_system.neighbourlist.particle_positions,
                                              trial_system.neighbourlist.cell_list,
                                              trial_system.neighbourlist.particle_neighbour_list)

            # calculate energy of trial system
            trial_system.energy = self._calculate_overall_energy()

            # evaluate system and trial system and append the accepted system to the trajectory
            self.opt_systems.append(
                MetropolisMonteCarlo.evaluate_trial_configuration_greedy(self.opt_systems[-1], trial_system))

        # interim update_radius
        update_radius = self.parameters.update_radius
        self.parameters.update_radius = update_radius/10

        # fine optimization
        for i in range(round(n_steps/10)):
            # generate trial system
            trial_system = MetropolisMonteCarlo.generate_trial_configuration(self.opt_systems[-1], self.parameters)

            # update particle positions and neighbourlist
            self.energy_calculator.set_system(trial_system.neighbourlist.particle_positions,
                                              trial_system.neighbourlist.cell_list,
                                              trial_system.neighbourlist.particle_neighbour_list)

            # calculate energy of trial system
            trial_system.energy = self._calculate_overall_energy()

            # evaluate system and trial system and append the accepted system to the trajectory
            self.opt_systems.append(
                MetropolisMonteCarlo.evaluate_trial_configuration_greedy(self.opt_systems[-1], trial_system))

        # resetting update_radius to desired value
        self.parameters.update_radius = update_radius

    def simulate(self, n_steps):

        # set up initial system from optimized system
        current_system = System(self.opt_systems[-1].particles, self.parameters)
        # pass particle positions and neighbourlist to the energy calculator class
        self.energy_calculator.set_system(current_system.neighbourlist.particle_positions,
                                          current_system.neighbourlist.cell_list,
                                          current_system.neighbourlist.particle_neighbour_list)
        # calculate energy of initial system
        current_system.energy = self._calculate_overall_energy()
        # append to trajectory
        self.sim_systems.append(current_system)

        for i in range(n_steps):

            # generate trial system
            trial_system = MetropolisMonteCarlo.generate_trial_configuration(self.sim_systems[-1], self.parameters)

            # update particle positions and neighbourlist
            self.energy_calculator.set_system(trial_system.neighbourlist.particle_positions,
                                              trial_system.neighbourlist.cell_list,
                                              trial_system.neighbourlist.particle_neighbour_list)

            # calculate energy of trial system
            trial_system.energy = self._calculate_overall_energy()

            # evaluate system and trial system and append the accepted system to the trajectory
            self.sim_systems.append(MetropolisMonteCarlo.evaluate_trial_configuration(self.sim_systems[-1], trial_system, self.parameters))

    # - - - private methods - - - #

    def _calculate_overall_energy(self):

        energy = Energy()
        short_ranged_energy = self.energy_calculator.calculate_shortranged_energy()
        energy.lj_energy = short_ranged_energy[0]
        energy.es_shortranged_energy = 0 #short_ranged_energy[1]
        energy.es_selfinteraction_energy = 0 #self.energy_calculator.calculate_selfinteraction_energy()
        energy.es_longranged_energy = 0 #self.energy_calculator.calculate_longranged_energy()

        energy.calculate_overall_energy()

        return energy
