import time
from collections import namedtuple
from l2l.optimizees.optimizee import Optimizee
from .pynn_network import Pynn_Net
import numpy as np
import random

MultiOptimizeeParameters = namedtuple(
    'MultiOptimizeeParameters', ['scale', 'nrec']) # TODO: add pre-sim-time, sim-time, dt? as parameters

class MultiOptimizee(Optimizee):
    def __init__(self, traj, parameters):
        super().__init__(traj)
        self.ind_idx = traj.individual.ind_idx
        self.generation = traj.individual.generation

        self.scale = parameters.scale
        self.nrec = parameters.nrec


    def create_individual(self):
        """
        Creates and returns a random individual
        """

        individual = {'weight_ex':  random.uniform(0     , 200),
                      'weight_in':  random.uniform(-1000  , 0),
                      'CE':         int(random.uniform(400     , 600)),
                      'CI':         int(random.uniform(50     , 150)),
                      'delay':      random.uniform(0.1   , 10),
                      }

        print("random individual:", individual)

        return individual


    def bounding_func(self, individual):
        """
        """
        # TODO what are reasonable bounds?
        # weight_ex         originally: JE_pA = 10.77                      now range: [1, 20]?   better [0, 200]
        # weight_in         originally: g*JE_pA = -5*10.77 = -53.85        now range: [-100, -5]? better [-1000, 0]
        # CE                originally: 9000 fixed                         now: pairwise bernoulli range: [0, 1]
        # CI                originally: 2250 fixed                         now: pairwise bernoulli range: [0, 1]
        # delay             originally: 1.5                                now range: [0.1, 10]

        individual = {'weight_ex':  np.clip(individual['weight_ex'] , 0     , 200),
                      'weight_in':  np.clip(individual['weight_in'] , -1000  , -0),
                      'CE':         np.clip(individual['CE']       , 400, 600),
                      'CI':         np.clip(individual['CI']       , 50, 150),
                      'delay':      np.clip(individual['delay']     , 0.1   , 10),
                      }
        return individual



    def simulate(self, traj):
        """
        """
        res = []
        self.ind_idx = traj.individual.ind_idx
        self.generation = traj.individual.generation

        for i in range(len(traj.individual.delay)):
            weight_ex = traj.individual.weight_ex[i]
            weight_in = traj.individual.weight_in[i]

            CE = int(traj.individual.CE[i])
            CI = int(traj.individual.CI[i])
            delay = traj.individual.delay[i]
            """net = Pynn_Net(scale=self.scale,
                                       CE=CE,
                                       CI=CI,
                                       weight_excitatory=weight_ex,
                                       weight_inhibitory=weight_in,
                                       delay=delay,
                                       nrec=self.nrec
                                       )"""
            net = Pynn_Net(scale=0.01,
                            CE=50,
                            CI=10,
                            weight_excitatory=15,
                            weight_inhibitory=-100,
                            delay=5,
                            nrec=5
                            )
            average_rate = net.run_simulation()

            desired_rate = 10
            fitness = -abs(average_rate - desired_rate) # TODO: is this a sensible way to calculate fitness?
            print(f'fitness {fitness} for ind {i}')
            res.append(fitness)
        print(f'fitnesses {res}')
        return res









