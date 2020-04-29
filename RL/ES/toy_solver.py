import cma
from time import sleep
from torch.multiprocessing import Process, Queue
from toy_worker import run_process

if __name__ == "__main__":
    NPARAMS = 100
    NPOPULATION = 101
    NUM_WORKERS = 20

    es = cma.CMAEvolutionStrategy(NPARAMS*[10],
                                1,
                                {'popsize': NPOPULATION,
                                })

    processes_queue = Queue()
    results_queue = Queue()
    done_queue = Queue()

    #start processes
    for wid in range(NUM_WORKERS):
        Process(target=run_process, args=(processes_queue, results_queue, done_queue)).start()

    while not es.stop():
        #start adding trial solutions to queue
        solutions = es.ask()
        for sol_id, sol in enumerate(solutions):
            processes_queue.put((sol_id, sol))

        #start processing results from run_process
        results = [0] * NPOPULATION
        for _ in range(NPOPULATION):
            while results_queue.empty():
                sleep(0.0001) #wait for results to come in
            sol_id, result = results_queue.get()
            results[sol_id] = result

        es.tell(solutions, results)
        es.disp()

    done_queue.put('Done')

    print("Local optimimum:", es.result.xbest)