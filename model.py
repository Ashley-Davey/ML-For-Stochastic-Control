import time
import tensorflow as tf
import numpy as np
from date import make_plots, log_run as log
from method import get_method






class Model:
    def __init__(self, config, method, monte):
        self.monte = monte


    def build(self, config, method, names):
        log("Building 2BSDE")

        self.bsde(config, method, names)
        log("BSDE init time: %d. Initialising Optimisers"%(time.time() - self.start_time))

        self.optimisers(config, method)


    def bsde(self, config, method, names):
        method.initialise_processes()
        with tf.variable_scope("forward", reuse=tf.AUTO_REUSE):

            for t in range(method.N):
                method.update_processes(t)
            method.makeBackwardsProcesses()
        method.get_losses()
        self.value_op = {}
        for name in names:
            self.value_op[name] = method.value_op(name, self.monte)

    def optimisers(self, config, method):
        self.global_step = tf.get_variable(
            "global_step",
            [],
            initializer=tf.constant_initializer(0),
            trainable=False,
            dtype=tf.int32,
        )

        apply_op_control = []
        apply_op_bsde = []
        control_loss = tf.zeros(1, dtype = tf.float64)
        self.control_grads = {}
        bsde_loss = tf.zeros(1, dtype = tf.float64)

        global_step_temp = self.global_step
        method.get_vars()
        self.loss_bsde = {}
        self.learning_rates = {}

        for name in method.names:

            decay_steps = config.decay_steps
            start, start_control = config.rate_start[name]
            learning_rate_control = tf.train.exponential_decay(
                start_control,
                self.global_step,
                decay_steps=decay_steps,
                decay_rate=0.1,
                staircase=True,
            )

            optimizer_control = tf.train.AdamOptimizer(learning_rate=learning_rate_control)
            learning_rate = tf.train.exponential_decay(
                start, self.global_step, decay_steps=decay_steps, decay_rate=0.1, staircase=True
            )

            self.learning_rates[name] = [learning_rate, learning_rate_control]
            optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)

            # now = time.time()
            variables = method.variables[name]['y']
            self.control_grads[name] = tf.zeros(1, dtype = tf.float64)
            self.loss_bsde[name] = tf.zeros(1, dtype = tf.float64)
            if variables is not None:
                grad_y = tf.gradients(method.loss_y[name], variables)
                apply_op_control.append(
                    optimizer_control.apply_gradients(
                        zip(grad_y, variables),
                        name="train_step_y_" + name,
                    )
                )
                self.control_grads[name] += tf.norm(grad_y)
            # print(time.time() - now)
            # now = time.time()
            for i in range( method.N):
                variables = method.variables[name]['control'][i]
                if variables is not None and len(variables) > 0:
                    grads = tf.gradients(method.loss_control[name][i], variables)
                    self.control_grads[name] += tf.reduce_mean([tf.norm(x) for x in grads]) / method.N
                    apply_op_control.append(
                        optimizer_control.apply_gradients(
                            zip(grads, variables),
                            global_step=global_step_temp,
                            name="train_step_control_" + str(i) + "_" + name,
                        )
                    )
                    global_step_temp = None
            # print(time.time() - now)
            # now = time.time()
            for i in range(method.N):
                variables = method.variables[name]['bsde'][i]
                if variables is not None and len(variables) > 0:
                    self.loss_bsde[name] += method.loss_bsde[name][i] / method.N
                    grads = tf.gradients(method.loss_bsde[name][i], variables)
                    # print(variables, grads)
                    bsde_loss += method.loss_bsde[name][i]
                    apply_op_bsde.append(
                        optimizer.apply_gradients(
                            zip(grads, variables),
                            global_step=global_step_temp,
                            name="train_step_bsde" + str(i) + "_" + name,
                        )
                    )
                    global_step_temp = None

            control_loss += self.control_grads[name]
            # print(time.time() - now)

        train_ops_control = apply_op_control + method._extra_train_ops_control
        train_op_control = tf.group(*train_ops_control)
        with tf.control_dependencies([train_op_control]):
            self.control_op = tf.identity(
                control_loss, name="train_op_control"
            )

        train_ops_bsde = apply_op_bsde + method._extra_train_ops_bsde
        train_op_bsde = tf.group(*train_ops_bsde)
        with tf.control_dependencies([train_op_bsde]):
            self.bsde_op = tf.identity(
                bsde_loss, name="train_op_bsde"
            )

        method.fixProcesses()





    def train(self, config, method, graph, names, verbose, sequential, markers, colors):

        operations = {}
        start_time = time.time()

        self.build(config, method, names)
        operations["value"] = self.value_op
        operations["bsde"] = self.loss_bsde
        operations["rates"] = self.learning_rates
        operations["step"] = self.global_step
        operations["control"] = self.control_grads
        
        self.CP = []

        # print( tf.get_default_graph().as_graph_def().node)
        

        with tf.Session(config=tf.ConfigProto(device_count={"GPU": 0})) as sess:

        # with tf.Session() as sess:
            log("Build time: %d. Training"%(time.time() - self.start_time))
            sess.run(tf.global_variables_initializer())

            if graph and not sequential:
                processes = sess.run(method.processes, feed_dict=method.sample(config.sample_size, rand = False))
                make_plots(processes, config, names, markers, colors, n = 'initial')


            if self.monte:
                log(f'Evaluating value using Monte Carlo with {config.repeats} repeats')



            if verbose:
                current_ops = sess.run(
                    operations,
                    feed_dict = method.sample(config.final_size)
                )

                for name in names:
                    self.values[name].append(np.array(current_ops["value"][name]))
                    if config.solution.soln:
                        self.rel_errors[name].append(
                            abs(self.values[name][-1] - config.solution.soln) / config.solution.soln
                        )
                    else:
                        self.rel_errors[name].append(1)

                    self.losses_bsde[name].append(current_ops["bsde"][name][0])
                    self.losses_control[name].append(current_ops["control"][name][0])

                    self.rates[name].append(current_ops["rates"][name])
                    self.steps[name].append(current_ops["step"])
                    self.running_times[name].append(time.time() - start_time)


                for name in names:
                    _value = f"{self.values[name][-1][0]:.5e} - {self.values[name][-1][1]:.5e}" if len(self.values[name][-1]) > 1 else f"{self.values[name][-1][0]:.5e}"
                    if config.solution.soln:
                        _error = f"{abs(self.rel_errors[name][-1][0])*100:.5e} - {abs(self.rel_errors[name][-1][1])*100:.5e}" if len(self.values[name][-1]) > 1 else f"{abs(self.rel_errors[name][-1][0])*100:.5e}"
                    else:
                        _error = f"{abs(self.rel_errors[name][-1])*100:.5e}"
                    log( f"{name}:\tvalue: {_value}\terror: {_error}\tbsde loss: {self.losses_bsde[name][-1]:.5e}\tcontrol loss: {self.losses_control[name][-1]:.5e}")

            operations["ops"] = [self.bsde_op, self.control_op]
            
            ##############################################################
            
            if 'primal' in names:
                new_processes = {
                    'X': method.processes['X']['primal'],
                    'primal_control': method.processes['primal_control']['primal']
                    }
            
                extra_processes = {}
                
                extra_processes['init'] = (
                    sess.run(
                        new_processes,
                        feed_dict=method.sample(config.sample_size_control, rand = False)
                    )
                )             
                
                
            ################################################################    
            
            
            max_iter = config.max_iter
            n_displaystep = config.display_steps
            
            log( "Optimising")
            try:
                
                


                # processes = sess.run([method.processes['Gamma2']], feed_dict = method.sample(2))
                # print(processes)
                # processes = sess.run([method.processes['Y']], feed_dict = method.sample(2))
                # print(processes)
                # processes = sess.run([method.processes['V2']], feed_dict = method.sample(2))
                # print(processes)
                # processes = sess.run([method.processes['dual_control']], feed_dict = method.sample(2))
                # print(processes)
                # processes = sess.run([method.processes], feed_dict = method.sample(2))
                # print(processes)
                # 1/0
                
                

                i = 0
                converged = False
                bsde_mean = {name: 1.0 for name in names}
                control_mean = {name: 1.0 for name in names}
                while i <= max_iter and not converged:
                    
                    pi, W = sess.run([
                        method.processes['primal_control'],
                        method.processes['W'],
                        ],
                        feed_dict = method.sample(1000)
                        )
                    self.CP.append({'pi': pi, 'W': W})
                    
                    feed_dict = method.sample(config.batch_size)



                    current_ops = sess.run(
                        operations,
                        feed_dict = feed_dict
                    )
                    # if i == 0:
                    #     feed_dict = method.sample(config.final_size)
                    #     g = sess.run(method.g_dual(method.processes['Y'][names[0]][method.N]), feed_dict = feed_dict)
                    #     Y = sess.run(method.processes['Y'][names[0]][method.N][:,:1], feed_dict = feed_dict)
                    #     J = sess.run(method.processes['J2'][names[0]][method.N], feed_dict = feed_dict)
                    #     Z = sess.run(method.Y_0[names[0]][0], feed_dict = feed_dict)
                    #     print(Z)
                    #     for j in range(config.final_size):
                    #         if J[j][0] > 100:
                    #             print(Y[j][0], g[j][0], J[j][0])

                        # 1/0

                    # feed_dict = method.sample(1)
                    # p = sess.run(method.processes, feed_dict = feed_dict)
                    # val = p['V1'][names[0]][0][0][0]
                    # der = p['Z1'][names[0]][0][0][0]
                    # inv = p['primal_control'][names[0]][0][0][0]
                    # con = p['primal_control'][names[0]][0][0][1]
                    # log(f'Step {i}: \t Value: {val:5e} \t der: {der:.5e} \t inv: {inv:.5e} \t con: {con:.5e}')

                    for name in names:
                        self.values[name].append(np.array(current_ops["value"][name]))
                        if config.solution.soln:
                            self.rel_errors[name].append(
                                abs(self.values[name][-1] - config.solution.soln) / abs(config.solution.soln)
                            )
                        else:
                            self.rel_errors[name].append(1)

                        self.losses_bsde[name].append(current_ops["bsde"][name][0])
                        self.losses_control[name].append(current_ops["control"][name][0])

                        self.rates[name].append(current_ops["rates"][name])
                        self.steps[name].append(current_ops["step"])
                        self.running_times[name].append(time.time() - start_time)

                    if i % n_displaystep == 0  and verbose:


    #                    if i in [0,9,49,99,499,999,1999,2999,3999,4999,5999,6999,7999,8999,9999]:
                        log("Step %5u\trunning time %5u\tlearning rate stage: %d"
                            % (
                                    i if i > 0 else 1,
                                    self.running_times[name][-1],
                                    ((max(i, 1) - 1) // max(config.decay_steps,1)) + 1
                                    )
                            )

                        value = sess.run(
                                    operations["value"],
                                    feed_dict = method.sample(config.final_size),
                                )
                        for name in names:
                            self.values[name][-1] = np.array(value[name])
                            if config.solution.soln:
                                self.rel_errors[name].append(
                                    abs(self.values[name][-1] - config.solution.soln) / abs(config.solution.soln)
                                )

                        for name in names:
                            _value = f"{self.values[name][-1][0]:.5e} - {self.values[name][-1][1]:.5e}" if len(self.values[name][-1]) > 1 else f"{self.values[name][-1][0]:.5e}"
                            if config.solution.soln:
                                _error = f"{abs(self.rel_errors[name][-1][0])*100:.5e} - {abs(self.rel_errors[name][-1][1])*100:.5e}" if len(self.values[name][-1]) > 1 else f"{abs(self.rel_errors[name][-1][0])*100:.5e}"
                            else:
                                _error = f"{abs(self.rel_errors[name][-1])*100:.5e}"
                            log( f"{name}:\t\tvalue: {_value}\terror(%): {_error}\tbsde loss: {self.losses_bsde[name][-1]:.5e}\tcontrol loss: {self.losses_control[name][-1]:.5e}")

                        if graph and not sequential:
                            processes = sess.run(method.processes, feed_dict=method.sample(config.sample_size, rand = False))

                            make_plots(processes, config, names, markers, colors, n = str(i))
                    
                            
                    if i > 200:
                        for name in names:
                            diff = 0
                        
                            losses = self.losses_bsde[name][max(i-50,0):]
                            bsde_mean[name] = abs(np.polynomial.polynomial.Polynomial.fit(range(len(losses)), losses, 1).convert().coef[-1] )
        
                            losses = self.losses_control[name][max(i-50,0):]
                            control_mean[name] = abs(np.polynomial.polynomial.Polynomial.fit(range(len(losses)), losses, 1).convert().coef[-1] )
                            
                            diff = max([diff, bsde_mean[name], control_mean[name]])
                        
                        converged = diff < config.tol
                        
                            
                    
                    i+= 1
            
                if i < max_iter:
                    log(f"Suspected plateau, terminated at step {i}")



            except KeyboardInterrupt:
                log(f"Manually disengaged at step {i}")
                
            

            # processes = sess.run([method.processes], feed_dict = method.sample(1))
            # print(processes)


            if self.monte:
                monte_names = names
            else:
                monte_names = [x for x in ['bruteprimal', 'brutedual', 'smp'] if x in names]

            if len(monte_names) > 0:
                log(f'Evalutating by monte carlo for {monte_names} with {config.repeats} repeats')

            for name in monte_names:
                self.values[name][-1] = 0

            for _ in range(config.repeats):
                value = sess.run(
                            {name: operations["value"][name] for name in monte_names},
                            feed_dict = method.sample(config.final_size),
                        )

                for name in monte_names:
                    # print(value[name])
                    self.values[name][-1] += np.array(value[name]) / config.repeats

            if config.solution.soln:
                self.rel_errors[name].append(
                    abs(self.values[name][-1] - config.solution.soln) / abs(config.solution.soln)
                )

            end_time = time.time()
            for key in method.randoms:
                method.processes[key] = getattr(method, key)


            self.final_processes = (
                sess.run(
                    method.processes,
                    feed_dict=method.sample(config.sample_size, rand = False)
                )
            )
            
            ###################################################################
            
            if 'primal' in names:
                new_processes = {
                    'X': method.processes['X']['primal'],
                    'primal_control': method.processes['primal_control']['primal']
                    }

                
                extra_processes['final'] = (
                    sess.run(
                        new_processes,
                        feed_dict=method.sample(config.sample_size_control, rand = False)
                    )
                )                 
                self.final_processes['extra'] = extra_processes 
            
            ###################################################################

            log("running time: %d" % (end_time - start_time))
            if config.solution.soln:
                log("soln:  %.8e" % config.solution.soln)
            for name in names:
                _value = f"{self.values[name][-1][0]:.5e} - {self.values[name][-1][1]:.5e}" if len(self.values[name][-1]) > 1 else f"{self.values[name][-1][0]:.5e}"
                if config.solution.soln:
                    _error = f"{abs(self.rel_errors[name][-1][0])*100:.5e} - {abs(self.rel_errors[name][-1][1])*100:.5e}" if len(self.values[name][-1]) > 1 else f"{abs(self.rel_errors[name][-1][0])*100:.5e}"
                else:
                    _error = f"{abs(self.rel_errors[name][-1])*100:.5e}"
                log( f"{name}:\tvalue: {_value}\terror(%): {_error}")

            # if 'primal' in method.names and 'dual' in method.names:
            #     log("Duaity gap: %.8e"%(100 * (self.values["dual"][-1] - self.values["primal"][-1]) / self.values["primal"][-1]))

            log("-" * 110)
            



















    def run(self, problem, config, verbose, graph, sequential, names, markers, colors):
        if self.monte or bool(set(['smp', 'bruteprimal', 'brutedual']) & set(names)):
            config.final_size = config.final
        else:
            config.final_size = 1
        self.values = {}
        self.rel_errors = {}
        self.losses_bsde = {}
        self.losses_control = {}
        self.running_times = {}
        self.steps = {}
        self.rates = {}
        for name in names:
            self.values[name] = []
            self.rel_errors[name] = []
            self.losses_bsde[name] = []
            self.losses_control[name] = []
            self.rates[name] = []
            self.running_times[name] = []
            self.steps[name] = []

        self.start_time = time.time()

        self.processes = {}


        if sequential:
            for name in names:
                log(f'Running {name}')
                tf.reset_default_graph()
                method = get_method(problem, config, [name], graph, self.monte)
                self.train(config, method, graph, [name], verbose, sequential)
                self.start_time = time.time()
                for p in self.final_processes.keys():
                    if p not in method.randoms:
                        if p not in self.processes.keys():
                            self.processes[p] = {}


                        n = 'common' if p == 'W' else name
                        self.processes[p][n] = self.final_processes[p][n]


        else:
            method = get_method(problem, config, names, graph, self.monte)
            self.train(config, method, graph, names, verbose, sequential, markers, colors)
            self.processes = self.final_processes











