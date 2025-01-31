import pickle
import sys
iteration = sys.argv[1]
idx = sys.argv[2]
handle_trajectory = open("/home/todt/Dokumente/L2L/results/L2L-SBI/simulation/trajectories/trajectory_" + str(iteration) + ".bin", "rb")
trajectory = pickle.load(handle_trajectory)
handle_trajectory.close()
handle_optimizee = open("/home/todt/Dokumente/L2L/results/L2L-SBI/simulation/optimizee.bin", "rb")
optimizee = pickle.load(handle_optimizee)
handle_optimizee.close()

trajectory.individual = trajectory.individuals[int(iteration)][int(idx)] 
res = optimizee.simulate(trajectory)

handle_res = open("/home/todt/Dokumente/L2L/results/L2L-SBI/simulation/results/results_" + str(iteration) + "_" + str(idx) + ".bin", "wb")
pickle.dump(res, handle_res, pickle.HIGHEST_PROTOCOL)
handle_res.close()

