import numpy as np
from l2l.utils.environment import Environment
from l2l.utils.experiment import Experiment

from l2l.optimizees.nest_hpc_benchmark import MultiOptimizee, MultiOptimizeeParameters
from l2l.optimizers.multigradientdescent.optimizer import MultiGradientDescentOptimizer
from l2l.optimizers.multigradientdescent.optimizer import MultiRMSPropParameters


def main():
    name = 'L2L-FUN-GD'
    experiment = Experiment("../results")
    traj, all_jube_params = experiment.prepare_experiment(name=name,
                                                          multiprocessing=False,
                                                          trajectory_name=name)

    # Optimizee
    optimizee_parameters = MultiOptimizeeParameters(scale=0.05,
                                                    nrec=100
                                                    )
    optimizee = MultiOptimizee(traj, optimizee_parameters)

    ## Outerloop optimizer initialization
    parameters = MultiRMSPropParameters(learning_rate=0.01, exploration_step_size=0.01,
                                   n_random_steps=1, momentum_decay=0.5, # TODO n_random_steps=1
                                   n_iteration=2, stop_criterion=np.Inf,
                                   seed=99, n_inner_params=4) # TODO n_inner_params = individuals!

    optimizer = MultiGradientDescentOptimizer(traj, optimizee_create_individual=optimizee.create_individual,
                                         optimizee_fitness_weights=(0.1,),
                                         parameters=parameters,
                                         optimizee_bounding_func=optimizee.bounding_func)

    experiment.run_experiment(optimizer=optimizer, optimizee=optimizee,
                              optimizer_parameters=parameters)

    experiment.end_experiment(optimizer)


if __name__ == '__main__':
    main()
