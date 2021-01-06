"""
Posthoc tests (to forward simulate RNN and find attractor types form trial conclusions and evaluate the saturation ration after the RNN is trained to perform SPM task)
written in Python 3.8.3
@ Elham
"""

import numpy as np
from SPM_task import *
#from matplotlib import pyplot as plt



def set_posthoc_params(x_ICs, r_ICs):
    ph_params = dict()
    ph_params['x_noise_var'] = 0.5 #determine type of attractor
    ph_params['n_fw'] = 50 # forward simulate the network (autonomously) for 50xTrial_length 
    ph_params['n_ICs'] = 100 # NO. of initial conditions for finding fps
    ph_params['seed'] = 1
    ph_params['ICs'] = {'r': r_ICs, 'x': x_ICs}. # ICs (x:state r:firing rate) saved after training
    ph_params['extend_delays'] = [5, 10, 15, 25, 35, 50, 100, 120, 150]
    #[5, 10, 15, 25, 50, 100, 150]
    ph_params['n_fw_end'] = 1  # fw from trial ends
    ph_params['failure_thresh'] = 0.1  # |z_nom - z_extend|
    ph_params['failure_type_thresh'] = 0.3  # |z_nom - z_settle|
    ph_params['n_trial'] = 10 # n noisy trials
    ph_params['noise_sigma'] = [10, 20, 50, 100]
    ph_params['stim_noise_mean'] = 0.0
    ph_params['stim_noise_sigma'] = ph_params['noise_sigma'][0]
    ph_params['n_off_list'] = [100, 200, 500, 700]
    #[500, 600, 700, 800, 900]
    ph_params['n_off'] = ph_params['n_off_list'][0]
    ph_params['sat_thresh'] = [-0.8, 0.8]

    return ph_params


def forward_simulate(params, initial_x, fw_steps):
    #fw simulate network, with no input from arbitrary initial condition
    # initail_x size= N*1

    net_prs = params['network']
    task_prs = params['task']
    model_prs = params['model']
    wo, wd = model_prs['wo'], model_prs['wd']
    wf, wfd, wi = model_prs['wf'], model_prs['wfd'], model_prs['wi']
    J = model_prs['J']
    dt, tau, g = net_prs['dt'], net_prs['tau'], net_prs['g']

    x = initial_x
    r = np.tanh(x)
    z = np.matmul(wo.T, r)
    zd = np.matmul(wd.T, r)

    x_mat, r_mat = np.zeros([net_prs['N'], fw_steps]), np.zeros([net_prs['N'], fw_steps])
    z_mat, zd_mat = np.zeros([net_prs['d_output'], fw_steps]),  np.zeros([net_prs['d_input'], fw_steps])

    for i in range(fw_steps):

        x_mat[:, i] = x.reshape(-1)
        r_mat[:, i] = r.reshape(-1)
        z_mat[:, i] = z
        zd_mat[:, i] = zd

        dx = -x + g * np.matmul(J, r) + np.matmul(wf, z) + np.matmul(wfd, zd)

        x = x + (dx * dt) / tau

        r = np.tanh(x)
        z = np.matmul(wo.T, r)
        zd = np.matmul(wd.T, r)

    return r_mat, x_mat, z_mat, zd_mat


def one_trial(params, ph_params, digits_rep, labels, trial, ic_index, extend=False):
    # takes ic_index eg 0 and a specific trial eg. (0,1) and returns trajectory of that trial

    net_prs = params['network']
    task_prs = params['task']
    model_prs = params['model']
    wo, wd = model_prs['wo'], model_prs['wd']
    wf, wfd, wi = model_prs['wf'], model_prs['wfd'], model_prs['wi']
    J = model_prs['J']
    dt, tau, g = net_prs['dt'], net_prs['tau'], net_prs['g']
    fw_steps = int(1 * task_prs['t_trial'] / net_prs['dt'])
    ph_params['delay_wm'] = ph_params['extend_delays'][0]
    ph_params['delay_out'] = 0


    task = sum_task_experiment(task_prs['n_digits'], 0, 0, 1, task_prs['time_intervals'],
                               net_prs['dt'], task_prs['output_encoding'], [trial], digits_rep, labels,
                               ph_params['seed'])

    exp_mat, target_mat, dummy_mat, i_digits, output_digits = task.experiment()


    x = ph_params['ICs']['x'][ic_index, :]
    r = np.tanh(x)
    z = np.matmul(wo.T, r)
    zd = np.matmul(wd.T, r)

    x_mat, r_mat = np.zeros([net_prs['N'], fw_steps]), np.zeros([net_prs['N'], fw_steps])


    for i in range(fw_steps):

        x_mat[:, i] = x.reshape(-1)
        r_mat[:, i] = r.reshape(-1)

        dx = -x + g * np.matmul(J, r) + np.matmul(wf, z) + np.matmul(wi, exp_mat[:, i].reshape(
            [net_prs['d_input'], 1])) + np.matmul(wfd, zd)


        x = x + (dx * dt) / tau
        r = np.tanh(x)
        z = np.matmul(wo.T, r)
        zd = np.matmul(wd.T, r)


    raug = None
    if extend:
        t_intervals = task_prs['time_intervals']
        cue_on = t_intervals['cue_on']
        stim_on, stim_off = t_intervals['stim_on'], t_intervals['stim_off']
        task_delay, response = t_intervals['delay_task'], t_intervals['response']

        x = ph_params['ICs']['x'][ic_index, :]
        r = np.tanh(x)
        z = np.matmul(wo.T, r)
        zd = np.matmul(wd.T, r)

        resp = (t_intervals['response'] + t_intervals['delay_task']) / dt
        aug1 = (t_intervals['fixate_on'] + t_intervals['fixate_off'] + t_intervals['cue_on'] + t_intervals['cue_off'] +
                t_intervals['stim_on'] + t_intervals['stim_off']) / dt
        aug2 = aug1 + (t_intervals['stim_on'] + t_intervals['stim_off']) / dt
        aug_out = aug2 + (t_intervals['delay_task'] + t_intervals['response']) / dt
        len_raug = ((task_prs['t_trial'] + ph_params['delay_wm'] + ph_params['delay_out']) * 1) / dt
        raug = np.zeros([net_prs['N'], int(len_raug)])

        r_counter = 0

        for i in range(fw_steps):

            if np.all(exp_mat[:, i] == 0.) and np.any(target_mat[:, i] == 0):

                dx = -x + g * np.matmul(J, r) + np.matmul(wf, z) + np.matmul(wi, exp_mat[:, i].reshape(
                    [net_prs['d_input'], 1])) + np.matmul(wfd, zd)
                x = x + (dx * dt) / tau


            else:

                dx = -x + g * np.matmul(J, r) + np.matmul(wf, z) + np.matmul(wi, exp_mat[:, i].reshape(
                    [net_prs['d_input'], 1])) + np.matmul(wfd, zd)
                x = x + (dx * dt) / tau


            r = np.tanh(x)
            z = np.matmul(wo.T, r)
            zd = np.matmul(wd.T, r)

            raug[:, r_counter] = r.reshape(-1)

            if i == aug1:

                d = int(ph_params['delay_wm'] / dt)
                while d > 0:
                    dx = -x + g * np.matmul(J, r) + np.matmul(wf, z) + np.matmul(wfd, zd)

                    x = x + (dx * dt) / tau

                    r = np.tanh(x)
                    z = np.matmul(wo.T, r)
                    zd = np.matmul(wd.T, r)

                    d -= 1
                    r_counter += 1
                    raug[:, r_counter] = r.reshape(-1)

            if i == aug_out or i == list(range(int(task_prs['t_trial'] * 1 / dt)))[-1]:

                d = int(ph_params['delay_out'] / dt)
                while d > 0:
                    dx = -x + g * np.matmul(J, r) + np.matmul(wf, z) + np.matmul(wfd, zd)
                    x = x + (dx * dt) / tau

                    r = np.tanh(x)
                    z = np.matmul(wo.T, r)
                    zd = np.matmul(wd.T, r)

                    d -= 1
                    r_counter += 1
                    raug[:, r_counter] = r.reshape(-1)

            r_counter += 1

    return r_mat, x_mat, raug


def fw_from_multi_ICs(params, ph_params, digits_rep, labels, trial, ic_index, ICtype='response', ic_zero=True):
    # forward simulate network from multiple ICs without giving network any input
    # ICs can be either during delay or end of each trial
    # ICtype='delay1' or 'delay2' or 'response', trial --> trial type eg (0,1) , ic_index--> IC type eg 0
    net_prs = params['network']
    train_prs = params['train']
    task_prs = params['task']
    msc_prs = params['msc']
    model_prs = params['model']
    wo, wd = model_prs['wo'], model_prs['wd']
    wf, wfd, wi = model_prs['wf'], model_prs['wfd'], model_prs['wi']
    J = model_prs['J']
    dt, tau, g = net_prs['dt'], net_prs['tau'], net_prs['g']
    fw_steps = int((ph_params['n_fw'] * task_prs['t_trial']) / net_prs['dt'])
    time_steps = np.arange(0, fw_steps, 1)
    t_intervals = task_prs['time_intervals']
    cue_on = t_intervals['cue_on']
    stim_on, stim_off = t_intervals['stim_on'], t_intervals['stim_off']
    task_delay, response = t_intervals['delay_task'], t_intervals['response']
    rng = np.random.RandomState(seed=ph_params['seed'])
    stim1 = cue_on + stim_on
    stim2 = stim1 + stim_off + stim_on

    if ICtype == 'delay1':
        _, x_mat, _ = one_trial(params, ph_params, digits_rep, labels, trial, ic_index, extend=False)
        initial_states = x_mat[:, int(stim1 / dt)].T
        # initial_states = x_mat[:, int(stim1/dt): int((stim1+stim_off)/dt)].T
        # index = rng.choice(initial_states.shape[0], ph_params['n_ICs'])
        # initial_states = initial_states[index, :]

        # ICs = np.zeros([1, net_prs['N']])
        # for i in range(initial_states.shape[0]):
        #     noise = rng.randn(1, initial_states.shape[1]) * ph_params['x_noise_var']
        #     ICnoise = initial_states[i, :] + noise
        #
        #     ICs = np.concatenate((ICs, ICnoise), axis=0)
        #
        # ICs = ICs[1:, :]
        ICs = initial_states[np.newaxis]


    elif ICtype == 'delay2':
        _, x_mat, _ = one_trial(params, ph_params, digits_rep, labels, trial, ic_index, extend=False)
        initial_states = x_mat[:, int(stim2 / dt): int((stim2 + stim_off) / dt)].T
        index = rng.choice(initial_states.shape[0], ph_params['n_ICs'])
        initial_states = initial_states[index, :]

        ICs = np.zeros([1, net_prs['N']])
        for i in range(initial_states.shape[0]):
            noise = rng.randn(1, initial_states.shape[1]) * ph_params['x_noise_var']
            ICnoise = initial_states[i, :] + noise

            ICs = np.concatenate((ICs, ICnoise), axis=0)

        ICs = ICs[1:, :]



    elif ICtype == 'response':
        initial_states = ph_params['ICs']['x'] #around fixed point -- shape  num_ICs X N
        initial_states = initial_states.reshape([initial_states.shape[0], initial_states.shape[1]])

        if ic_zero == False:
            initial_states = initial_states.reshape([initial_states.shape[0], initial_states.shape[1]])

        elif ic_zero == True:
            initial_states = np.vstack((np.zeros([1, net_prs['N']]), initial_states))  # shape = (5, 1000)

        # initial_states = i_params['ICs']['x'][IC, :][np.newaxis]



        ICs = np.zeros([1, net_prs['N']])
        for i in range(initial_states.shape[0]):

            IC_mat = np.zeros([ph_params['n_ICs'] // initial_states.shape[0], net_prs['N']])
            for j in range(ph_params['n_ICs']//initial_states.shape[0]):
                noise = rng.randn(1, initial_states.shape[1]) * ph_params['x_noise_var']
                IC_mat[j, :] = initial_states[i,:] + noise

            ICs = np.concatenate((ICs, IC_mat), axis=0)


        ICs = ICs[1:, :]


    trajectories = np.zeros([fw_steps, net_prs['N'], ph_params['n_ICs']])
    z_mat, zd_mat = np.zeros([fw_steps, net_prs['d_output'], ph_params['n_ICs']]), np.zeros([fw_steps, net_prs['d_input'], ph_params['n_ICs']])
    # fw simulate from set of initial conditions for i_params['n_fw'] steps

    for k in range(ICs.shape[0]):

        x = ICs[k, :].T
        r = np.tanh(x)
        z = np.matmul(wo.T, r)
        zd = np.matmul(wd.T, r)

        x_mat, r_mat = np.zeros([net_prs['N'], fw_steps]), np.zeros([net_prs['N'], fw_steps])
        z_, zd_ = np.zeros([net_prs['d_output'], fw_steps]), np.zeros([net_prs['d_input'], fw_steps])


        for i in range(fw_steps):

            x_mat[:, i] = x.reshape(-1)
            r_mat[:, i] = r.reshape(-1)
            z_[:, i] = z
            zd_[:, i] = zd

            dx = -x + g * np.matmul(J, r) + np.matmul(wf, z) + np.matmul(wfd, zd)

            x = x + (dx * dt) / tau

            r = np.tanh(x)
            z = np.matmul(wo.T, r)
            zd = np.matmul(wd.T, r)


        trajectories[:, :, k] = r_mat.T
        z_mat[:, :, k] = z_.T
        zd_mat[:, :, k] = zd_.T

    return trajectories, z_mat, zd_mat


def attractor_type(params, ph_params, digits_rep, labels):
    # evaluate type of attractors using 1. mean and variance of z and z_d 2. evaluation of unique fixed points given
    #  multiple noisy ics

    output_encoding = params['task']['output_encoding']
    t_trial = params['task']['t_trial']
    dt = params['network']['dt']
    steps = 2 * int(t_trial/dt)
    trajectories, z_mat, zd_mat = fw_from_multi_ICs(params, ph_params, digits_rep, labels, (0,0), 0, ICtype='response', ic_zero=True)
    z_variance = np.var(z_mat[-steps:, :, :], axis=0).astype(np.float32)

    z_mean = np.mean(z_mat[-steps:, :, :], axis=0).astype(np.float32)
    zd_mean = np.mean(zd_mat[-steps:, :, :], axis=0).astype(np.float32)
    num_fps = len(np.unique(z_mat[-1:, :, :].round(4)))

    attractor = 'unknown'
    if np.all(z_variance < 10e-10):
        if num_fps >= 2 * len(output_encoding):
            attractor = 'fixedPoint' + str(num_fps)

        elif num_fps < 2 * len(output_encoding):
            attractor = 'manifoldFps' + str(num_fps)

    else:
        if np.unique(z_mean).shape[0] == z_mat.shape[2]:
            attractor = 'limitCycle'
        elif np.max(z_mean) > max(output_encoding) + 2.5 or np.min(z_mean) < -max(output_encoding) - 2.5:
            attractor = 'far_someFPs'
        else:
            attractor = 'both'


    return trajectories, np.unique(z_mean), np.unique(zd_mean), attractor



def attractor_type_nozero(params, ph_params, digits_rep, labels):
    # evaluate type of attractors using 1. mean and variance of z and z_d 2. evaluation of unique fixed points given
    #  multiple noisy ics

    output_encoding = params['task']['output_encoding']
    t_trial = params['task']['t_trial']
    dt = params['network']['dt']
    steps = 2 * int(t_trial/dt)
    ph_params['x_noise_var'] = 0.0
    ph_params['n_ICs'] = len(params['task']['keep_perms'])
    trajectories, z_mat, zd_mat = fw_from_multi_ICs(params, ph_params, digits_rep, labels, (0,0), 0, ICtype='response', ic_zero=False)
    z_variance = np.var(z_mat[-steps:, :, :], axis=0).astype(np.float32)

    z_mean = np.mean(z_mat[-steps:, :, :], axis=0).astype(np.float32)
    zd_mean = np.mean(zd_mat[-steps:, :, :], axis=0).astype(np.float32)
    num_fps = len(np.unique(z_mat[-1:, :, :].round(2)))

    attractor = 'unknown'
    if np.all(z_variance < 10e-7):
        if num_fps >=  len(output_encoding):
            attractor = 'fixedPoint' + str(num_fps)

        elif num_fps <  len(output_encoding):
            attractor = 'manifoldFps' + str(num_fps)

    else:
        if np.unique(z_mean).shape[0] == z_mat.shape[2]:
            attractor = 'limitCycle'
        elif np.max(z_mean) > max(output_encoding) + 2.5 or np.min(z_mean) < -max(output_encoding) - 2.5:
            attractor = 'far_someFPs'
        else:
            attractor = 'both'


    return trajectories, np.unique(z_mean), np.unique(zd_mean), attractor



def asymp_trial_end(params, ph_params):
    # fw simulate from trial ends, to evaluate how deep the fixed points are, no noise
    ph_params['n_fw'] = 50
    ph_params['n_ICs'] = len(params['task']['keep_perms'])
    net_prs = params['network']
    task_prs = params['task']
    model_prs = params['model']
    wo, wd = model_prs['wo'], model_prs['wd']
    wf, wfd, wi = model_prs['wf'], model_prs['wfd'], model_prs['wi']
    J = model_prs['J']
    dt, tau, g = net_prs['dt'], net_prs['tau'], net_prs['g']
    fw_steps = int((ph_params['n_fw'] * task_prs['t_trial']) / net_prs['dt'])
    trial_length = int(task_prs['t_trial'] / net_prs['dt'])
    time_points = [0*trial_length, 1*trial_length, 5*trial_length, 10*trial_length, 20*trial_length, 30*trial_length,
                   40*trial_length, 50*trial_length-1 ]
    all_norm_F = np.zeros([ph_params['n_ICs'], len(time_points)])

    initial_states = ph_params['ICs']['x']  # around fixed point -- shape  num_ICs X N
    initial_states = initial_states.reshape([initial_states.shape[0], initial_states.shape[1]])

    ICs = initial_states

    trajectories = np.zeros([fw_steps, net_prs['N'], ph_params['n_ICs']])
    z_mat, zd_mat = np.zeros([fw_steps, net_prs['d_output'], ph_params['n_ICs']]), np.zeros(
        [fw_steps, net_prs['d_input'], ph_params['n_ICs']])
    # fw simulate from set of initial conditions for i_params['n_fw'] steps

    for k in range(ICs.shape[0]):

        norm_F = []
        x = ICs[k, :].T
        r = np.tanh(x)
        z = np.matmul(wo.T, r)
        zd = np.matmul(wd.T, r)

        x_mat, r_mat = np.zeros([net_prs['N'], fw_steps]), np.zeros([net_prs['N'], fw_steps])
        z_, zd_ = np.zeros([net_prs['d_output'], fw_steps]), np.zeros([net_prs['d_input'], fw_steps])

        for i in range(fw_steps):

            x_mat[:, i] = x.reshape(-1)
            r_mat[:, i] = r.reshape(-1)
            z_[:, i] = z
            zd_[:, i] = zd

            dx = -x + g * np.matmul(J, r) + np.matmul(wf, z) + np.matmul(wfd, zd)

            x = x + (dx * dt) / tau

            if i in time_points:
                norm_F.append(np.linalg.norm(dx))

            r = np.tanh(x)
            z = np.matmul(wo.T, r)
            zd = np.matmul(wd.T, r)

        trajectories[:, :, k] = r_mat.T
        z_mat[:, :, k] = z_.T
        zd_mat[:, :, k] = zd_.T
        all_norm_F[k,:] = np.array(norm_F).T

    return all_norm_F


def vf_delays(params, ph_params, digits_rep, labels):

    # evaluate the value of output from the start of first delay (how vector field is used to encode delays)
    # returns all_traj: size= fw_steps X N X trials X ic_num , all_z and all_zd

    ph_params['n_fw'] = 20
    ph_params['n_ICs'] = 1
    fw_steps = int((ph_params['n_fw'] * params['task']['t_trial']) / params['network']['dt'])
    trials = params['task']['keep_perms']
    all_traj = np.zeros([fw_steps, params['network']['N'], 1])
    all_z, all_zd = np.zeros([fw_steps, params['network']['d_output'], 1]), np.zeros([fw_steps, params['network']['d_input'], 1])

    for ic_index in range(len(trials)):
        for trial in trials:
            trajectories, z_mat, zd_mat = fw_from_multi_ICs(params, ph_params, digits_rep, labels,
                                                            trial, ic_index, ICtype='delay1', ic_zero=True)

            all_traj = np.concatenate((all_traj, trajectories), axis=2)
            all_z = np.concatenate((all_z, z_mat), axis=2)
            all_zd = np.concatenate((all_zd, zd_mat), axis=2)

    all_traj, all_z, all_zd = all_traj[:, :, 1:], all_z[:, :, 1:], all_zd[:, :, 1:]

    return all_traj, all_z, all_zd


def extend_nominal_delays(params, ph_params, digits_rep, labels):
    #extend first delay, and evaluate the failure type (does it even fail? asymptotic or fixed_time)
    # returns all_fail : shape= len(all_delayes) X 16
    trials =  params['task']['keep_perms']
    fw_steps = int((ph_params['n_fw_end']* params['task']['t_trial'])/ params['network']['dt'])
    wo, wd = params['model']['wo'], params['model']['wd']
    all_fail = []

    for d, delay in enumerate(ph_params['extend_delays']):
        ph_params['delay_wm'] = delay
        ph_params['delay_out'] = 0
        fail = []

        for trial in trials:
            for ic_index in range(len(trials)):

                r_mat ,_ , r_extend = one_trial(params, ph_params, digits_rep, labels,
                                                             trial, ic_index, extend=True)

                zo_nominal = np.matmul(wo.T, r_mat[:,-1])
                zd_nominal = np.matmul(wd.T, r_mat[:,-1])

                zo_ext = np.matmul(wo.T, r_extend[:,-1])
                zd_ext = np.matmul(wd.T, r_extend[:, -1])

                initial_x = np.arctanh(r_extend[:, -1])
                r_settle, x_settle, z_settle, zd_settle = forward_simulate(params, initial_x, fw_steps)


                evaluate_failure_type(zo_nominal, zo_ext, z_settle[:, -1], ph_params, fail)


        all_fail.append(fail)

    return all_fail



def saturation_percentage(params, ph_params, digits_rep, labels):

    trials = params['task']['keep_perms']
    t_intervals = params['task']['time_intervals']
    cue_on = t_intervals['cue_on']
    stim_on, stim_off = t_intervals['stim_on'], t_intervals['stim_off']
    task_delay, response = t_intervals['delay_task'], t_intervals['response']
    dt = params['network']['dt']
    N = params['network']['N']
    r_delay = np.zeros([N, int(stim_off/dt), len(trials)**2])
    r_resp = np.zeros([N, int(response/dt), len(trials)**2])
    delay_time = cue_on + stim_on
    resp_time = delay_time + stim_off + stim_on + stim_off + task_delay
    c = 0

    for ic_index in range(len(trials)):
        for trial in trials:
            r_mat, x_mat, _ = one_trial(params, ph_params, digits_rep, labels, trial, ic_index, extend=False)
            r_delay[:, :, c] = r_mat[:, int(delay_time/dt): int((delay_time+stim_off)/dt)]
            r_resp[:, :, c] = r_mat[:, int(resp_time/dt):]

            c += 1



    r_delay_mean = np.mean(r_delay, axis=1)
    r_resp_mean = np.mean(r_resp, axis=1)
    sat_ratio_delay = [len(np.where((r_delay_mean[:, i]> ph_params['sat_thresh'][1]) | (r_delay_mean[:, i]< ph_params['sat_thresh'][0]))[0])/N
             for i in range(c)]
    sat_ratio_resp = [len(
        np.where((r_resp_mean[:, i] > ph_params['sat_thresh'][1]) | (r_resp_mean[:, i] < ph_params['sat_thresh'][0]))[
            0])/N for i in range(c)]


    return sat_ratio_delay, sat_ratio_resp








































