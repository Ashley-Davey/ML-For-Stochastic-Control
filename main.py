# -*- coding: utf-8 -*-
"""
SGD Methods for Stochastic Control

"""
import time
import tensorflow as tf
from config import get_config
from model import Model
from date import log, log_run, init, make_plots
import numpy as np
from argparse import ArgumentParser
from matplotlib import pyplot as plt, rcParams
import json

rcParams['figure.dpi'] = 600


class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, np.float32) or isinstance(obj, np.float64):
            return float(obj)
        return json.JSONEncoder.default(self, obj)

def run_DCBSDE(params, verbose, graph, sequential,  names, markers, colors, monte):
    tf.reset_default_graph()
    config = get_config(problem, **params)
    

    
    log(f"T = {config.T:.1f}, N = {config.N}, init = {config.Xinit[0]}")
    try:
        log(f"c: {config.c}, gamma: {config.gamma}")
    except AttributeError:
        pass
    if config.solution.soln:
        log(f"Solution: {config.solution.soln:.5e}" )
    log(f'Control Space: {config.space}')

    date = init()
    config.dump()

    model = Model(problem, config, monte)

    model.run(
        problem, config, verbose, graph, sequential, names, markers, colors
    )

    with open( date + '/results.json', 'w') as f:
        json.dump({
                'values': model.values,
                'losses_bsde': model.losses_bsde,
                'losses_control': model.losses_control,
                'rates': model.rates,
                'rel_errors': model.rel_errors,
                'running_times': model.running_times
                }, f, indent = 4, cls = NumpyEncoder)

    with open(date + '/final.txt', 'a') as f:
        for name in names:
            f.write(f'{config.N} \t {config.T} \t {config.k} \t {name} \t {model.values[name][-1]} \t {model.rel_errors[name][-1]} \t {model.running_times[name][-1]} \n')

    processes = model.processes

    log_run('Saving data figures')

    plots = [
        ['values',"Value Approximation", 'linear'],
        [  'losses_bsde','BSDE Loss', 'log'],
        ['losses_control','Control Loss', 'log'],
        ['rel_errors', 'Relative Error', 'log'],
    ]

    for item, ylabel, yscale in plots:
        fig, axes = plt.subplots(1)
        axes.grid(alpha=0.5)
        for name in names:
            axes.plot(getattr(model, item)[name], color = colors[name], label = name.capitalize())
        axes.set_ylabel(ylabel)
        axes.set_xlabel('Iteration Step')

        if item == 'values' and  config.solution.soln:
            axes.axhline(config.solution.soln, color="r", label = 'Solution')
        axes.set_yscale(yscale)
        if item == 'rel_errors':
            axes.set_ylim(1e-8,1.0)

        axes.legend()
        plt.savefig(f'{date}/{item}.png',bbox_inches='tight')
        plt.close()


    if graph:
        log('saving process figures')
        make_plots(processes, config, names, markers, colors)
        
        

    
    return {'config': config, 'model': model}



if __name__ == "__main__":
    log("Started at", time.strftime("%H:%M:%S"))
    parser = ArgumentParser(description="DCBSDE Methods")

    defined_methods = ['primal', 'dual', 'smp', 'bruteprimal', 'brutedual', 'hybrid', 'pde']
    markers = {'primal': 'x', 'dual': '+', 'smp': '1', 'bruteprimal': '2', 'brutedual': '3', 'hybrid': '4', 'pde': 'o'}
    colors = {'primal': 'g', 'dual': 'b', 'smp': 'm', 'bruteprimal': 'c', 'brutedual': 'k', 'hybrid': 'y', 'pde': 'C4'}


    for method in defined_methods:
        assert markers[method] and colors[method], f'missing color/marker for {method}'
        parser.add_argument("--" + method, action = "store_true")

    parser.add_argument("--problem"     , "-P"                       )
    parser.add_argument("--quiet"       , "-q", action = "store_true")
    parser.add_argument("--graph"       , "-g", action = "store_true")
    parser.add_argument("--sequential"  , "-s", action = "store_true")
    arguments = parser.parse_args()
    problem = arguments.problem

    names = []
    for method in defined_methods:
        if getattr(arguments, method):
            names += [method]

    verbose = not arguments.quiet
    graph = arguments.graph
    sequential = arguments.sequential
    monte = False

    if len(names) == 1:
        sequential = False #override
    if verbose:
        log(f"Solving {problem} with: {names}")
    if sequential:
        log('running each method seperately')
        
    #####################################################################
        
    params = {
        'space': 'whole',
        'N': 10,
        }

    tf.reset_default_graph()
    results = run_DCBSDE(params, verbose, graph, sequential,  names, markers, colors, monte)




