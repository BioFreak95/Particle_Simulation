import numpy as np


class Parameters:

    cell_shift_list = np.array([
        [0, 1, -1, 1, -1, 1, -1, 0, 0, 0, 1, -1, 1, -1, 1, -1, 0, 0, 0, 1, -1, 1, -1, 1, -1, 0, 0],
        [0, 0, 0, 1, 1, -1, -1, 1, -1, 0, 0, 0, 1, 1, -1, -1, 1, -1, 0, 0, 0, 1, 1, -1, -1, 1, -1],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, -1, -1, -1, -1, -1, -1, -1, -1, -1]])

    def __init__(self, temperature, box, es_sigma, update_radius, update_probability, charges, lj_sigmas, lj_epsilons,
                 accuracy=10, cutoff=0):
        if temperature < 0:
            raise ValueError('Temperature cant be negative.')
        for i in box:
            if i < 0:
                raise ValueError('boxsize can not be negative.')
        if es_sigma < 0:
            raise ValueError('es_sigma can not be negative.')
        if update_radius < 0:
            raise ValueError('update_radius can not be negative.')

        self.temperature = temperature
        self.box = box
        self.es_sigma = es_sigma
        self.update_radius = update_radius
        self.update_probability = update_probability
        self.accuracy = accuracy

        if cutoff == 0:
            self.cutoff_radius = self.calc_cutoff()
        else:
            self.cutoff_radius = cutoff

        self.K_cutoff = self.calc_Kcutoff()
        self.k_vector = self.calc_kvector()

        self.charges = charges
        self.lj_sigmas = lj_sigmas
        self.lj_epsilons = lj_epsilons

    def calc_3Dkvector(self):
        k_vectors = []
        for i in range(int(np.floor(-self.K_cutoff)), int(np.ceil(self.K_cutoff + 1))):
            for j in range(int(np.floor(-self.K_cutoff)), int(np.ceil(self.K_cutoff + 1))):
                for l in range(int(np.floor(-self.K_cutoff)), int(np.ceil(self.K_cutoff + 1))):
                    k_vector = [i, j, l]
                    if np.linalg.norm(k_vector) <= self.K_cutoff:
                        vec = []
                        for m in range(len(k_vector)):
                            vec.append(k_vector[m])
                        for k in range(len(vec)):
                            vec[k] *= -1
                        if vec in k_vectors:
                            continue
                        else:
                            k_vectors.append(k_vector)
        k_vectors.remove([0, 0, 0])
        return k_vectors

    def calc_2Dkvector(self):
        k_vectors = []
        for i in range(int(np.floor(-self.K_cutoff)), int(np.ceil(self.K_cutoff + 1))):
            for j in range(int(np.floor(-self.K_cutoff)), int(np.ceil(self.K_cutoff + 1))):
                k_vector = [i, j]
                if np.linalg.norm(k_vector) <= self.K_cutoff:
                    vec = []
                    for m in range(len(k_vector)):
                        vec.append(k_vector[m])
                    for k in range(len(vec)):
                        vec[k] *= -1
                    if vec in k_vectors:
                        continue
                    else:
                        k_vectors.append(k_vector)
        k_vectors.remove([0, 0])
        return k_vectors

    def calc_1Dkvector(self):
        k_vectors = []
        for i in range(int(np.floor(-self.K_cutoff)), int(np.ceil(self.K_cutoff + 1))):
            k_vector = [i]
            if np.linalg.norm(k_vector) <= self.K_cutoff:
                vec = []
                for m in range(len(k_vector)):
                    vec.append(k_vector[m])
                for k in range(len(vec)):
                    vec[k] *= -1
                if vec in k_vectors:
                    continue
                else:
                    k_vectors.append(k_vector)
        k_vectors.remove([0])
        return k_vectors

    def calc_kvector(self):
        dim = len(self.box)
        k_vectors = []
        if dim == 3:
            k_vectors = self.calc_3Dkvector()
        if dim == 2:
            k_vectors = self.calc_2Dkvector()
        if dim == 1:
            k_vectors = self.calc_1Dkvector()
        return np.array(k_vectors)

    def calc_cutoff(self):
        boxlength = max(self.box)
        cutoff = 0.49 * boxlength
        return cutoff

    def calc_Kcutoff(self):
        cutoff = (2 * self.accuracy) / self.cutoff_radius
        return cutoff