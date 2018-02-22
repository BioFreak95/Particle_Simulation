import numpy as np


class Parameters:

    cell_shift_list = np.array([
        [0, 1, -1, 1, -1, 1, -1, 0, 0, 0, 1, -1, 1, -1, 1, -1, 0, 0, 0, 1, -1, 1, -1, 1, -1, 0, 0],
        [0, 0, 0, 1, 1, -1, -1, 1, -1, 0, 0, 0, 1, 1, -1, -1, 1, -1, 0, 0, 0, 1, 1, -1, -1, 1, -1],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, -1, -1, -1, -1, -1, -1, -1, -1, -1]])

    def __init__(self, temperature, box, es_sigma,  charges, lj_sigmas, lj_epsilons,
                 update_probability=0.75, update_radius=0, accuracy=10, cutoff=0):
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
        self.update_probability = update_probability
        self.accuracy = accuracy

        if update_radius == 0:
            self.update_radius = self._calculate_update_radius()
        else:
            self.update_radius = update_radius

        if cutoff == 0:
            self.cutoff_radius = self._calculate_cutoff()
        else:
            self.cutoff_radius = cutoff

        self.K_cutoff = self._calculate_K_cutoff()
        self.k_vector = self._calculate_k_vector()

        self.charges = charges
        self.lj_sigmas = lj_sigmas
        self.lj_epsilons = lj_epsilons

    def _calculate_3D_k_vector(self):
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

    def _calculate_2D_k_vector(self):
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

    def _calculate_1D_k_vector(self):
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

    def _calculate_k_vector(self):
        dim = len(self.box)
        k_vectors = []
        if dim == 3:
            k_vectors = self._calculate_3D_k_vector()
        if dim == 2:
            k_vectors = self._calculate_2D_k_vector()
        if dim == 1:
            k_vectors = self._calculate_1D_k_vector()
        return np.array(k_vectors)

    def _calculate_cutoff(self):
        boxlength = max(self.box)
        cutoff = 0.5 * boxlength
        return cutoff

    def _calculate_K_cutoff(self):
        cutoff = (2 * self.accuracy) / self.cutoff_radius
        return cutoff

    def _calculate_update_radius(self):
        update_radius = np.sum(self.box)/(len(self.box) * 20)
        return update_radius