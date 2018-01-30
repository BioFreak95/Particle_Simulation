class Energy:
    def __init__(self):
        self.overall_energy = None
        self.lj_energy = None
        self.es_energy = None
        self.es_shortranged_energy = None
        self.es_longranged_energy = None
        self.es_selfinteraction_energy = None

    def calculate_overall_energy(self):

        if self.lj_energy is None:
            raise ValueError("Some energy values are not defined. Overall energy can only be computed when all sub "
                             "energies are available.")
        elif self.es_energy is None:
            Energy.calculate_es_energy(self)

        self.overall_energy = self.lj_energy + self.es_energy

    def calculate_es_energy(self):

        if self.es_longranged_energy is None or \
                        self.es_shortranged_energy is None or \
                        self.es_selfinteraction_energy is None:
            raise ValueError("Some energy values are not defined. Overall energy can only be computed when all sub "
                             "energies are available.")
        else:
            self.es_energy = self.es_longranged_energy + self.es_shortranged_energy - self.es_selfinteraction_energy
