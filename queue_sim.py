#!/usr/bin/env python3

import argparse
import csv
import json
import logging
import numpy as np
import supermarket_plot as plotting
import collections
from random import sample, seed
from matplotlib import pyplot as plt
from workloads import weibull_generator # for weibull behaviour instead of memoryless behaviour
from discrete_event_sim import Simulation, Event

# One possible modification is to use a different distribution for job sizes or and/or interarrival times.
# Weibull distributions (https://en.wikipedia.org/wiki/Weibull_distribution) are a generalization of the
# exponential distribution, and can be used to see what happens when values are more uniform (shape > 1,
# approaching a "bell curve") or less (shape < 1, "heavy tailed" case when most of the work is concentrated
# on few jobs).

# To use Weibull variates, for a given set of parameter do something like
# from workloads import weibull_generator
# gen = weibull_generator(shape, mean)
#
# and then call gen() every time you need a random variable


# columns saved in the CSV file
CSV_COLUMNS = ['lambd', 'mu', 'max_t', 'n', 'd', 'w']


# Monitoring for generating the chart
class Monitoring(Event):
    
    def process(self, sim):
        # registering all queues seen length        
        for i in range(sim.n):
            sim.length_counter[sim.queue_len(i)] += 1
        sim.num_samples += sim.n    
            
        #scheduling the next monitoring event
        sim.schedule(sim.monitor_interval, Monitoring())
        
        
        

class Queues(Simulation):
    '''
        n => servers count
        lambda => Jobs arrive rate time
        mu => server serve time
        d => random queues(server) choosing 
        
        weibull_shape_service/arrival => for weibull simulation(non exponential behaviour)
    '''
    """Simulation of a system with n servers and n queues.

    The system has n servers with one queue each. Jobs arrive at rate lambd and are served at rate mu.
    When a job arrives, according to the supermarket model, it chooses d queues at random and joins
    the shortest one.
    """
    

    def __init__(self, lambd, mu, n, d, weibull_shape_service, weibull_shape_arrival, queue_capacity=None,
        overflow_behaviour_option="drop", max_retries=3):
        
        super().__init__()
        self.running = [None] * n  # if not None, the id of the running job (per queue)
        self.queues = [collections.deque() for _ in range(n)]  # FIFO queues of the system
        # NOTE: we don't keep the running jobs in self.queues
        self.arrivals = {}  # dictionary mapping job id to arrival time
        self.completions = {}  # dictionary mapping job id to completion time
        self.lambd = lambd
        self.n = n
        self.d = d
        self.mu = mu
        self.arrival_rate = lambd * n # frequency of new jobs is proportional to the number of queues
        
        # extension
        self.queue_capacity = queue_capacity
        self.overflow_behaviour_option = overflow_behaviour_option
        self.max_retries = max_retries
        
        self.total_arrivals = 0
        self.dropped_jobs = 0
        self.retry_attemps = 0
         
        self.weibull_shape_service = weibull_shape_service
        if weibull_shape_service <= 0 or weibull_shape_arrival <= 0:
            raise ValueError("Weibull shape parameters must be greater than 0")
        
        mean_service = 1 / mu
        self.weibull_service_gen = weibull_generator(self.weibull_shape_service,mean_service)
            
        self.weibull_shape_arrival = weibull_shape_arrival
        mean_interarrival = 1 / self.arrival_rate
        self.weibull_arrival_gen = weibull_generator(self.weibull_shape_arrival,mean_interarrival) 
        
        # schedule the first arrival
        self.schedule_arrival(0)
        
        self.length_counter = collections.Counter()
        self.num_samples = 0
        self.monitor_interval = 1
        
        self.schedule(self.monitor_interval,Monitoring())
        
    def schedule_arrival(self, job_id):
        """Schedule the arrival of a new job."""
        
        delay = self.weibull_arrival_gen()
        self.schedule(delay, Arrival(job_id))


    def schedule_completion(self, job_id, queue_index): 
        """Schedule the completion of a job."""

        job_service_time = self.weibull_service_gen()
        # schedule the time of the completion event
        self.schedule(job_service_time,Completion(job_id,queue_index))
                    

    def queue_len(self, i):
        """Return the length of the i-th queue.
        
        Notice that the currently running job is counted even if it is not in self.queues[i]."""

        return (self.running[i] is not None) + len(self.queues[i])


class Arrival(Event):
    """Event representing the arrival of a new job."""

    def __init__(self, job_id):
        self.id = job_id

    def process(self, sim: Queues):
        
        sim.total_arrivals += 1
        #طبق اصل supermarket باید d تا صف رو رندوم انتخاب کنیم
        sample_queues = sample(range(0,sim.n),sim.d)
        queue_index = min(sample_queues,key=sim.queue_len)
        
        accepted = True
        
        if sim.queue_capacity is not None and sim.queue_len(queue_index) >= sim.queue_capacity:
            accepted = False
            
            if (sim.overflow_behaviour_option == "retry"):
                 # chosing d sample queues untill max_retires
                for _ in range(sim.max_retries):
                    sim.retry_attemps += 1
                    sample_queues = sample(range(0,sim.n),sim.d)
                    queue_index = min(sample_queues,key=sim.queue_len)
                    if(sim.queue_len(queue_index) < sim.queue_capacity):
                        accepted = True
                        break
            if not accepted:
                sim.dropped_jobs += 1
        
        if accepted:
            sim.arrivals[self.id] = sim.t
            # if there is no running job in the queue:
                # set the incoming one
                # schedule its completion
            # otherwise, put the job into the queue
            if(sim.running[queue_index] is None):
                sim.running[queue_index] = self.id
                sim.schedule_completion(self.id,queue_index)
            else:
                sim.queues[queue_index].append(self.id)
            
        sim.schedule_arrival(self.id + 1)
        

class Completion(Event):
    """Job completion."""

    def __init__(self, job_id, queue_index):
        self.job_id = job_id  # currently unused, might be useful when extending
        self.queue_index = queue_index

    def process(self, sim: Queues):
        queue_index = self.queue_index
        assert sim.running[queue_index] == self.job_id  # the job must be the one running
        sim.completions[self.job_id] = sim.t
        queue = sim.queues[queue_index]
        if queue:  # queue is not empty
            sim.running[queue_index] = new_job_id = queue.popleft()  # assign the first job in the queue
            sim.schedule_completion(new_job_id, queue_index)  # schedule its completion
        else:
            sim.running[queue_index] = None  # no job is running on the queue


def main():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--lambd', type=float, default=0.7, help="arrival rate")
    parser.add_argument('--mu', type=float, default=1, help="service rate")
    parser.add_argument('--max-t', type=float, default=1_000_000, help="maximum time to run the simulation")
    parser.add_argument('--n', type=int, default=1, help="number of servers")
    parser.add_argument('--d', type=int, default=1, help="number of queues to sample")
    parser.add_argument('--csv', help="CSV file in which to store results")
    parser.add_argument('--plot', action="store_true", help="plot combined curves after simulation")
    parser.add_argument('--plot_only', action='store_true', help="only plot existing d*.json files without running a simulation")
    parser.add_argument("--seed",type=int, help="random seed")
    parser.add_argument("--verbose", action='store_true')   
    parser.add_argument("--out", type=str, help="output filename for CDF json file" ) 
    parser.add_argument("--weibull_shape_service", default=1.0, type=float, help="enter the shape value for weibull")       
    parser.add_argument("--weibull_shape_arrival", default=1.0, type=float, help="weibull_shape arrival rate (k = 1 => exponential, k < 1 => heavy-tail)")   
    #extension
    parser.add_argument("--queue_capacity", type=int, default=None, help="setting max queue capacity (including running job). If not set, queues are infinite.")    
    parser.add_argument("--overflow_behaviour_option", choices=["drop", "retry"], default="drop", help="Options when the queue is full.")
    parser.add_argument("--max_retries", type=int, default=3, help= "Max retries if overflow_behaviour_option is retry.")
    args = parser.parse_args()


    params = [getattr(args, column) for column in CSV_COLUMNS[:-1]]
    # corresponds to params = [args.lambd, args.mu, args.max_t, args.n, args.d]

    if any(x <= 0 for x in params):
        logging.error("lambd, mu, max-t, n and d must all be positive")
        exit(1)

    if args.seed:
        seed(args.seed)  # set a seed to make experiments repeatable
    if args.verbose:
        # output info on stderr
        logging.basicConfig(format='{levelname}:{message}', level=logging.INFO, style='{')

    if args.lambd >= args.mu:
        logging.warning("The system is unstable: lambda >= mu")
    
    
    # checking the d variable ( must be greated than n and positive for sure)    
    if args.d <= 0:
        logging.error("d must be a positive integer")
        exit(1)

    if args.d > args.n:
        logging.error(f"d must be <= n (got d={args.d}, n={args.n})")
        exit(1)    
        
        
    
    def cdf_files():
        import glob 
        import re
        
        files = glob.glob("d*.json")  
        d_values = []
        mapping = {}
        
        for filename in files:
            match = re.match(r"d(\d+)\.json", filename)
            if match:
                d = int(match.group(1))
                d_values.append(d)
                mapping[d] = filename
                
        d_values.sort()
        return d_values, mapping        
        
    if(args.plot_only):
        d_values, mapping = cdf_files()
        plotting.plot_combined(d_values, args.lambd, mapping)
        return    
        
        
        
    sim = Queues(args.lambd,
                 args.mu,
                 args.n,
                 args.d,
                 args.weibull_shape_service,
                 args.weibull_shape_arrival,
                 args.queue_capacity,
                 args.overflow_behaviour_option,
                 args.max_retries)
    
    sim.run(args.max_t)

    completions = sim.completions
    if len(sim.completions)  > 0:
        W = ((sum(completions.values()) - sum(sim.arrivals[job_id] for job_id in completions))
         / len(completions))
        print(f"Average time spent in the system: {W}")
    else:
        W = None
        print("Average time spent in the system: None (there are no completed jobs)")
    
    
    if sim.queue_capacity is not None:
        drop_rate = sim.dropped_jobs/sim.total_arrivals if sim.total_arrivals > 0 else 0.0
        print(f"Dropped jobs: {sim.dropped_jobs} / {sim.total_arrivals} => drop rate is: {drop_rate:.4f}")
        if sim.overflow_behaviour_option == "retry":
            print(f"Total retry attempt is: {sim.retry_attemps}")
   
    count = sim.length_counter
    total = sim.num_samples
    max_len =  max(count.keys()) if total > 0 else 0
    
    cdf = {} 
    running = 0
    for x in range(max_len, -1, -1):
        running += count.get(x,0)
        cdf[x] = running / total

    
   

    # for storing the cdfs as json
    if args.out:
        with open(args.out, "w") as f:
            json.dump(cdf,f)
            
    if(args.plot):
        d_values, mapping = cdf_files()
        plotting.plot_combined(d_values, args.lambd, mapping)
        
        
    if args.csv is not None:
        with open(args.csv, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(params + [W])
    
        
if __name__ == '__main__':
    main()
