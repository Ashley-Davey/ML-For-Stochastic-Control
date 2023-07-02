import time
import os
from matplotlib import pyplot as plt, rcParams
from matplotlib.lines import Line2D
import numpy as np
rcParams['figure.dpi'] = 600

date = None

def init():
    try:
        os.mkdir("data")
    except FileExistsError:
        pass
    global date
    date = time.strftime("data/%Y_%m_%d-%H_%M_%S")
    os.mkdir(date)
    return date

def log(*args, **kwargs):
    now = time.strftime("%H:%M:%S")
    print("[" + now + "]: ", end="")
    print(*args, **kwargs)

def log_run(*args, **kwargs):
    global date

    now = time.strftime("%H:%M:%S")
    print("[" + now + "]: ", end="")
    print(*args, **kwargs)
    with open(date + '/run.txt', 'a') as f:
        print("[" + now + "]: ", end="", file = f)
        print(*args, **kwargs, file = f)

def log_config(*args, **kwargs):
    global date
    with open(date + '/config.txt', 'a') as f:
        print(*args, **kwargs, file = f)

def get_date():
    global date
    return date

def make_plots(processes, config, names, markers, colors, n = 'final'):

    try:
        try:
            os.mkdir(date + '/processes/' + n)
        except FileNotFoundError:
            os.mkdir(date + '/processes')
            os.mkdir(date + '/processes/' + n)
        h = config.h
        paths = config.sample_size
        legend_elements = []

        for name in names:
            legend_elements.append(
                Line2D(
                    [0],
                    [0],
                    marker=markers[name],
                    color=colors[name],
                    label=name.capitalize(),
                    )
                )

        if config.solution.soln:
            legend_elements.append(
                Line2D([0], [0], color="r", label="Solution", linestyle="dashed")
            )

        plots = [x for x in ['X', 'Z1', 'V1', 'primal_control', 'dual_control', 'Gamma1', 'V', 'Yb', 'Vb', 'Zb', 'Xp', 'Vp', 'Zp', 'Gp'] if x in processes.keys()]

        for process in plots:
            path = processes[process]
            fig, axes = plt.subplots(1)
            axes.grid(alpha = 0.5)
            taxis = np.arange(config.N-1)
            for i in range(paths):
                for name in path.keys():
                    taxis = np.arange(len(path[name]))
                    # print(process, name, [(path[name][t][i] if (len(path[name][t][i].shape) < 2) else path[name][t][i][0][0]) for t in taxis] )
                    axes.plot(
                        taxis * h, [(path[name][t][i] if (len(path[name][t][i].shape) < 2) else path[name][t][i][0,:]) for t in taxis], color=colors[name], marker=markers[name]
                    )
                    # axes.plot(
                    #     taxis * h, [(path[name][t][i] if (len(path[name][t][i].shape) < 2) else path[name][t][i][:,0]) for t in taxis], color=colors[name], marker=markers[name]
                    # )
                    # if process == 'X' and 'Y' in processes.keys() and name in processes['Y'].keys():
                    #     axes.plot(
                    #         taxis * h, [- np.power(processes['Y'][name][t][i][0], config.q) / config.q for t in taxis], linestyle = 'dashed'
                    #     )
                    if process + '_real' in processes.keys() and name in processes[process + '_real'].keys():
                        path2 = processes[process + '_real'][name]
                        axes.plot(
                            taxis * h, [(path2[t][i] if (len(path2[t][i].shape) < 2) else path2[t][i][0,:]) for t in taxis], color=colors[name], marker=markers[name], linestyle = 'dashdot'
                        )
            try:
                if getattr(config.solution, process) is not None:
                    for i in range(paths):
                        axes.plot(
                            taxis * h,
                            [
                                getattr(config.solution, process)(t * h, [processes['W']['common'][s][i] for s in range(t + 1)])
                                if (len(getattr(config.solution, process)(t * h, [processes['W']['common'][s][i] for s in range(t + 1)]).shape) < 2)
                                else getattr(config.solution, process)(t * h, [processes['W']['common'][s][i] for s in range(t + 1)])[:,0]
                                for t in taxis
                            ],
                            color="r",
                            linestyle="dashed",
                        )
            except Exception as e:
                log(f'Cant plot {process} solution : {e}')
            axes.set_ylabel(f"{process.replace('_',' ')}")
            axes.set_xlabel("Time")
            axes.legend(handles=legend_elements)
            plt.savefig(f'{date}/processes/{n}/{process}.png',bbox_inches='tight')
            plt.close()
    except KeyboardInterrupt:
        log("Manually disengaged plotting")
