import cma
from time import time, sleep
import argparse
import os
import pickle
import numpy as np
import torch
from torch.multiprocessing import Process, Queue
import config
from rollout import RolloutGenerator
from model import Controller

SIGMA_INIT = 0.1
MODEL_DIR = 'models'

def run_process(processes_queue, results_queue, done_queue, env_name):
    """
    Each process gets parameters from the queue to use in the rollout and puts results in a queue
    """
    generator = RolloutGenerator(env_name)
    while done_queue.empty():
        if processes_queue.empty():
            sleep(0.1) #wait for queue to fill
        else:
            sol_id, sol = processes_queue.get()
            results_queue.put((sol_id, generator.rollout(sol)))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--env', type=str, help="minitaur, racecar")
    parser.add_argument('--retrain', type=bool, default=False, help="Start new instead of loading existing model (T/F)")
    parser.add_argument('--target', type=int, help="Target score")
    parser.add_argument('--average-over', type=int, help="Maintain average of target for the mean of population over past x iterations")
    parser.add_argument('--save-every', type=int, help="Save stats and model every x iterations")
    parser.add_argument('--n-samples', type=int, help="Number of samples to estimate average reward from each set of parameters")
    parser.add_argument('--pop-size', type=int, help="Population size")
    parser.add_argument('--max-workers', type=int, help="Number of workers")
    args = parser.parse_args()

    n_samples = args.n_samples
    pop_size = args.pop_size
    num_workers = min(args.max_workers, n_samples * pop_size)

    env_prop = config.envs[args.env]
    controller = Controller(env_prop['INPUT_DIM'], env_prop['HIDDEN_DIMS'], env_prop['ACTION_DIM'], env_prop['ACTIVATION'])
    existing_path = os.path.join(MODEL_DIR, "{}_es.pkl".format(args.env))
    if args.retrain or not os.path.exists(existing_path):
        es = cma.CMAEvolutionStrategy(current_params, SIGMA_INIT, {'popsize': pop_size})
    else: #if don't retrain and model exists, load that model
        es = pickle.load(open(os.path.join(MODEL_DIR, "{}_es.pkl".format(args.env)), 'rb'))

    processes_queue = Queue()
    results_queue = Queue()
    done_queue = Queue()

    results_summary = []
    current_best = None
    current_params = controller.flat_parameters() #start with random
    iteration = 0

    start_time = time()
    #start processes
    for wid in range(num_workers):
        Process(target=run_process, args=(processes_queue, results_queue, done_queue, args.env)).start()

    while not es.stop():
        #start adding trial solutions to queue
        solutions = es.ask()
        for sol_id, sol in enumerate(solutions):
            for _ in range(n_samples):
                processes_queue.put((sol_id, sol))

        #start processing results from run_process
        results = [0] * pop_size
        for _ in range(pop_size * n_samples):
            while results_queue.empty():
                sleep(0.1) #wait for results to come in
            sol_id, result = results_queue.get()
            results[sol_id] += result / n_samples #average across different rollouts

        results_summary.append(((sum(results)/len(results)), min(results), max(results)))

        es.tell(solutions, results)
        es.disp()

        iteration += 1

        if iteration % args.save_every == 0:

            with open(os.path.join(MODEL_DIR, "{}_summary.pkl".format(args.env)), 'wb') as f:
                    pickle.dump(results_summary, f)

            if (current_best == None) or (es.result.fbest > current_best):
                current_best = es.result.fbest
                current_params = es.result.xfavorite #save mean of current population
                controller.set_parameters(current_params)
                torch.save({
                    'iteration': iteration,
                    'reward': current_best,
                    'model_state_dict': controller.state_dict(),
                    }, os.path.join(MODEL_DIR, "{}_iter.tar".format(args.env)))
                #save es state
                pickle.dump(es, open(existing_path, 'wb'))
            else:
                pass #don't save until params are found that's better

        if (np.mean(np.array(results_summary)[-args.average_over:,0])) < -args.target:
            break

    done_queue.put('Done')
    time_elapsed = time() - start_time

    print("Took %d seconds" % time_elapsed)
    controller.set_parameters(es.result.xbest)
    torch.save({
                    'iteration': iteration,
                    'reward': current_best,
                    'model_state_dict': controller.state_dict(),
                    }, os.path.join(MODEL_DIR, "{}_final.tar".format(args.env)))

    #python solver.py --env racecar --target -25 --average-over 10 --save-every 5 --n-sample 16 --pop-size 8 --max-workers 16
    #python solver.py --env minitaur --target 13 --average-over 10 --save-every 10 --n-sample 16 --pop-size 8 --max-workers 16