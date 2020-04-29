import cma
from time import sleep

def run_process(processes_queue, results_queue, done_queue):
    fit_func = cma.fitness_functions.FitnessFunctions().rastrigin
    while done_queue.empty():
        if processes_queue.empty():
            sleep(0.0001) #wait for queue to fill
        else:
            sol_id, sol = processes_queue.get()
            results_queue.put((sol_id, fit_func(sol)))