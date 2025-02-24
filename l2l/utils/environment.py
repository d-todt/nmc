from l2l.utils.trajectory import Trajectory
#from l2l.utils.JUBE_runner import JUBERunner
import logging


logger = logging.getLogger("utils.environment")


class Environment:
    """
    The Environment class takes the place of the pypet Environment and provides the required functionality
    to execute the inner loop. This means it uses either JUBE or sequential calls in order to execute all
    individuals in a generation.
    Based on the pypet environment concept: https://github.com/SmokinCaterpillar/pypet
    """

    def __init__(self, *args, **keyword_args):
        """
        Initializes an Environment
        :param args: arguments passed to the environment initialization
        :param keyword_args: arguments by keyword. Relevant keywords are trajectory and filename.
        The trajectory object holds individual parameters and history per generation of the exploration process.
        """
        if 'trajectory' in keyword_args:
            self.trajectory = Trajectory(name=keyword_args['trajectory'], debug = keyword_args['debug'],
                                         stop_run = keyword_args['stop_run'], timeout=keyword_args['timeout'])
        if 'checkpoint' in keyword_args:
            self.trajectory = keyword_args["checkpoint"]
            self.trajectory.is_loaded = True
        if 'filename' in keyword_args:
            self.filename = keyword_args['filename']
        self.postprocessing = None
        self.multiprocessing = True
        if 'multiprocessing' in keyword_args:
            self.multiprocessing = keyword_args['multiprocessing']
        self.run_id = 0
        self.enable_logging()

    def run(self, runfunc):
        """
        Runs the optimizees using either JUBE or sequential calls.
        :param runfunc: The function to be called from the optimizee
        :return: the results of running a whole generation. Dictionary indexed by generation id.
        """
        result = {}
        logger.info("Environment start iteration")
        for it in range(self.trajectory.individual.generation, self.trajectory.par['n_iteration']+self.trajectory.individual.generation):
            logger.info(f"Environment run generation {it+1}/{self.trajectory.par['n_iteration']}")
            if self.multiprocessing:
                raise NotImplementedError('No JUBE!')

            else:
                # Sequential calls to the runfunc in the optimizee
                logger.info('runner sequential')
                result[it] = []
                # Call runfunc on each individual from the trajectory
                try:
                    for ind in self.trajectory.individuals[it]:
                        self.trajectory.individual = ind
                        result[it].append((ind.ind_idx, runfunc(self.trajectory)))
                        self.run_id = self.run_id + 1
                        logger.info(f'run id {ind}')
                except:
                    if self.logging:
                        logger.exception("Error during serial execution of individuals")
                    raise
            # Add results to the trajectory
            self.trajectory.results.f_add_result_to_group("all_results", it, result[it])
            self.trajectory.current_results = result[it]
            # Update trajectory file
            #jube.dump_traj(self.trajectory) # TODO
            # Perform the postprocessing step in order to generate the new parameter set
            self.postprocessing(self.trajectory, result[it])
        return result

    def add_postprocessing(self, func):
        """
        Function to add a postprocessing step
        :param func: the function which performs the postprocessing. Postprocessing is the step where the results
        are assessed in order to produce a new set of parameters for the next generation.
        """
        self.postprocessing = func

    def enable_logging(self):
        """
        Function to enable logging
        TODO think about removing this.
        """
        self.logging = True

    def disable_logging(self):
        """
        Function to enable logging
        """
        self.logging = False
