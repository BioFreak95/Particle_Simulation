import numpy as np


class Parameters:

    def __init__(self, temperature, box, es_sigma, update_radius, particle_types, cutoff_radius, K_cutoff):
        self.temperature = temperature
        self.box = box
        self.es_sigma = es_sigma
        self.update_radius = update_radius
        self.particle_types = particle_types
        self.cutoff_radius = cutoff_radius
        self.K_cutoff = K_cutoff
        self.k_vector = self.calc_kvector()

    def calc_kvector(self):
        k_vectors = []
        for i in range(-self.K_cutoff, self.K_cutoff + 1):
            for j in range(-self.K_cutoff, self.K_cutoff + 1):
                for l in range(-self.K_cutoff, self.K_cutoff + 1):
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
