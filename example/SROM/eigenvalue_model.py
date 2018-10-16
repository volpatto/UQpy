import numpy as np
import sys


class RunPythonModel:

    def __init__(self, samples=None, dimension=None):

        self.samples = samples
        self.dimension = dimension
        self.QOI = np.zeros_like(self.samples)
        for i in range(self.samples.shape[0]):
            p = np.array([[self.samples[i, 0]+self.samples[i, 1], -self.samples[i, 1], 0], 
                          [-self.samples[i, 1], self.samples[i, 1]+self.samples[i, 2], -self.samples[i, 2]], 
                          [0, -self.samples[i, 2], self.samples[i, 2]]])
            w, v = np.linalg.eig(p)
            self.QOI[i, :] = w

    # index = sys.argv[1]
    # filename = 'modelInput_{0}.txt'.format(int(index))
    # x = np.loadtxt(filename, dtype=np.float32)
    #
    # p = np.sqrt(abs(np.sum(x)))
    #
    # with open('solution_{0}.txt'.format(int(index)), 'w') as f:
    #     f.write('{} \n'.format(p))
