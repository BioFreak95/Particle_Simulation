class ParticleType:
    def __init__(self, name, mass, charge, lj_epsilon, lj_sigma):
        if not isinstance(name, str):
            raise TypeError('name must be a string!')
        if mass < 0:
            raise ValueError('mass can not be negative!')
        if charge < 0:
            raise ValueError('charge can not be negative!')
        if lj_epsilon < 0:
            raise ValueError('lj epsilon can not be negative!')
        if lj_sigma < 0:
            raise ValueError('lj sigma can not be negative!')

        self.name = name
        self.mass = mass
        self.charge = charge
        self.lj_epsilon = lj_epsilon
        self.lj_sigma = lj_sigma
