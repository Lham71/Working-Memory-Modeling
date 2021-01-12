'''
based on the identified category of each trained model
1. load each model parameter
2. extend delay with high precision (persistence)
3. distraction (robuustness to noise)
this script should be isolated from other scripts

written in Python 3.8.3
@ Elham
'''

import argparse
import pickle
from SPM_task import *


indir = '/scratch/elham/results3500c/'   #directory containing trained models parameters
outdir = '/scratch/elham/Tests3500cNoiseCont/'   # output directory for saving results

parser = argparse.ArgumentParser()
parser.add_argument('-n', '--name', nargs='+', type=str)
arg = parser.parse_args()
file_name = arg.name[0]

with open(indir + 'train_' + file_name, 'rb') as f:
    params, x_ICs, r_ICs, error_ratio = pickle.load(f)


def set_test_params(x_ICs, r_ICs):

    '''set test parameters which are similar to ph_params but reassigned with different values,
    they are renamed to avoid bugs
    returns: dictionary of test parameters
    '''
    test_params = dict()
    test_params['seed'] = 1
    test_params['ICs'] = {'r': r_ICs, 'x': x_ICs}
    test_params['extend_delays'] = np.arange(1,200,0.5)
    test_params['n_fw_end'] = 1  # forward simulate network from trial ends
    test_params['failure_thresh'] = 0.1  # |z_norm - z_extend|
    test_params['failure_type_thresh'] = 0.3  # |z_norm - z_settle|
    test_params['n_trial'] = 10 # n noisy trials
    test_params['noise_sigma'] = [0.1, 0.2, 0.25,0.33, 0.5, 1, 2,3, 4,5, 6,7, 8,9, 10, 15, 20, 25, 30, 35, 40 ] # start from 0 and small steps
    test_params['stim_noise_mean'] = 0.5
    test_params['n_off_list'] = [100, 200, 300, 400, 500, 600, 700, 800, 900]

    return test_params


def get_digits_reps():

    with open('allDigCNNMNIST', 'rb') as f:
        z_mean, z_log_var, z_sample, x_test, y_test = pickle.load(f)

    y_test, x_test = np.array(y_test), x_test.reshape([x_test.shape[0], 28, 28])

    return y_test, z_sample


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


def one_trial(params, test_params, digits_rep, labels, trial, ic_index, extend=False, delay_wm=0):
    # takes ic_index eg 0 and a specific trial eg. (0,1) and returns trajectory of that trial

    net_prs = params['network']
    task_prs = params['task']
    model_prs = params['model']
    wo, wd = model_prs['wo'], model_prs['wd']
    wf, wfd, wi = model_prs['wf'], model_prs['wfd'], model_prs['wi']
    J = model_prs['J']
    dt, tau, g = net_prs['dt'], net_prs['tau'], net_prs['g']
    fw_steps = int(1 * task_prs['t_trial'] / net_prs['dt'])
    test_params['delay_out'] = 0

    task = sum_task_experiment(task_prs['n_digits'], 0, 0, 1, task_prs['time_intervals'],
                               net_prs['dt'], task_prs['output_encoding'], [trial], digits_rep, labels,
                               test_params['seed'])

    exp_mat, target_mat, dummy_mat, i_digits, output_digits = task.experiment()


    x = test_params['ICs']['x'][ic_index, :]
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


    raug, xaug = None, None

    if extend:
        if delay_wm == 0:
            raise ValueError(
                'extend delay is {}'.format(delay_wm))

        t_intervals = task_prs['time_intervals']
        cue_on = t_intervals['cue_on']
        stim_on, stim_off = t_intervals['stim_on'], t_intervals['stim_off']
        task_delay, response = t_intervals['delay_task'], t_intervals['response']

        x = test_params['ICs']['x'][ic_index, :]
        r = np.tanh(x)
        z = np.matmul(wo.T, r)
        zd = np.matmul(wd.T, r)

        resp = (t_intervals['response'] + t_intervals['delay_task']) / dt
        aug1 = (t_intervals['fixate_on'] + t_intervals['fixate_off'] + t_intervals['cue_on'] + t_intervals['cue_off'] +
                t_intervals['stim_on'] + t_intervals['stim_off']) / dt
        aug2 = aug1 + (t_intervals['stim_on'] + t_intervals['stim_off']) / dt
        aug_out = aug2 + (t_intervals['delay_task'] + t_intervals['response']) / dt
        len_raug = ((task_prs['t_trial'] + delay_wm + test_params['delay_out']) * 1) / dt
        raug = np.zeros([net_prs['N'], int(len_raug)])
        xaug = np.zeros([net_prs['N'], int(len_raug)])

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
            xaug[:, r_counter] = x.reshape(-1)

            if i == aug1:

                d = int(delay_wm / dt)
                while d > 0:
                    dx = -x + g * np.matmul(J, r) + np.matmul(wf, z) + np.matmul(wfd, zd)

                    x = x + (dx * dt) / tau

                    r = np.tanh(x)
                    z = np.matmul(wo.T, r)
                    zd = np.matmul(wd.T, r)

                    d -= 1
                    r_counter += 1
                    raug[:, r_counter] = r.reshape(-1)
                    xaug[:, r_counter] = x.reshape(-1)



            if i == aug_out or i == list(range(int(task_prs['t_trial'] * 1 / dt)))[-1]:

                d = int(test_params['delay_out'] / dt)
                while d > 0:
                    dx = -x + g * np.matmul(J, r) + np.matmul(wf, z) + np.matmul(wfd, zd)
                    x = x + (dx * dt) / tau

                    r = np.tanh(x)
                    z = np.matmul(wo.T, r)
                    zd = np.matmul(wd.T, r)

                    d -= 1
                    r_counter += 1
                    raug[:, r_counter] = r.reshape(-1)
                    xaug[:, r_counter] = x.reshape(-1)

            r_counter += 1

    return r_mat, x_mat, raug, xaug


def evaluate_failure_type(zo_target, zo_corr, zo_settle, ph_params, fail):


    if abs(zo_target - zo_corr) > ph_params['failure_thresh']:

        if abs(zo_target - zo_settle) > ph_params['failure_type_thresh']:
            fail.append((True, 'asymptotic'))
        else:
            fail.append((True, 'fixed_time'))
    else:

        fail.append((False, 'notFail'))

    return fail


def add_noise_to_stim(exp_mat, params, test_params, seed):

    # add noise to non zero elements of exp_mat (only for first stimulus)
    task_prs = params['task']
    dt = params['network']['dt']
    t_intervals = task_prs['time_intervals']
    stim_on = int(t_intervals['stim_on']/dt)

    mu = test_params['stim_noise_mean']
    noise_sig = test_params['stim_noise_sigma']
    rng = np.random.RandomState(seed)
    noise = rng.rand(params['network']['d_input'], stim_on) * noise_sig + mu
    tmp = exp_mat[:, :stim_on] + noise
    noisy_exp_mat = np.hstack((tmp, exp_mat[:,stim_on:]))


    return noisy_exp_mat


def extend_nominal_delays(params, test_params, digits_rep, labels):
    #extend first delay, and evaluate the failure type (does it even fail? asymptotic or fixed_time)
    # returns all_fail : shape= len(all_delayes) X 16

    trials = params['task']['keep_perms']
    fw_steps = int((test_params['n_fw_end'] * params['task']['t_trial']) / params['network']['dt'])
    wo, wd = params['model']['wo'], params['model']['wd']
    all_fail = []

    delay_trial_ic = np.zeros([len(test_params['extend_delays']), len(trials), len(trials)], dtype=object)

    for d, delay in enumerate(test_params['extend_delays']):

        fail = []

        for tr, trial in enumerate(trials):
            for ic_index in range(len(trials)):

                r_mat ,_ , r_extend, x_extend = one_trial(params, test_params, digits_rep, labels,
                                                             trial, ic_index, extend=True, delay_wm=delay)

                zo_nominal = np.matmul(wo.T, r_mat[:,-1])
                zd_nominal = np.matmul(wd.T, r_mat[:,-1])

                zo_ext = np.matmul(wo.T, r_extend[:,-1])
                zd_ext = np.matmul(wd.T, r_extend[:, -1])

                #initial_x = np.arctanh(r_extend[:, -1])
                initial_x = x_extend[:, -1]
                r_settle, x_settle, z_settle, zd_settle = forward_simulate(params, initial_x, fw_steps)

                fail = evaluate_failure_type(zo_nominal, zo_ext, z_settle[:, -1], test_params, fail)


                delay_trial_ic[d, tr, ic_index] = ' '.join([str(delay), str(trial), str(ic_index)])



        all_fail.append(fail)

    return all_fail, delay_trial_ic


def extend_nominal_delays_error(params, test_params, digits_rep, labels):

    # extend first delay, and evaluate the failure type (does it even fail? asymptotic or fixed_time)
    # returns all_fail : shape= len(all_delayes) X 16

    trials = params['task']['keep_perms']
    fw_steps = int((test_params['n_fw_end'] * params['task']['t_trial']) / params['network']['dt'])
    wo, wd = params['model']['wo'], params['model']['wd']


    delay_trial_ic = np.zeros([len(test_params['extend_delays']), len(trials), len(trials)], dtype=object)
    all_error_out = np.zeros([len(test_params['extend_delays']), len(trials)**2, 3, 1])
    all_error_dummy = np.zeros([len(test_params['extend_delays']), len(trials) ** 2, 3,2])


    for d, delay in enumerate(test_params['extend_delays']):

        all_zo, all_zd = [], []

        for tr, trial in enumerate(trials):
            for ic_index in range(len(trials)):

                r_mat, _, r_extend, x_extend = one_trial(params, test_params, digits_rep, labels, trial, ic_index, extend=True, delay_wm=delay)

                zo_nominal = np.matmul(wo.T, r_mat[:,-1])
                zd_nominal = np.matmul(wd.T, r_mat[:,-1])


                zo_ext = np.matmul(wo.T, r_extend[:,-1])
                zd_ext = np.matmul(wd.T, r_extend[:, -1])

                # initial_x = np.arctanh(r_extend[:, -1])
                initial_x = x_extend[:, -1]
                r_settle, x_settle, z_settle, zd_settle = forward_simulate(params, initial_x, fw_steps)

                z_t_c_s = (zo_nominal, zo_ext, z_settle[:, -1])
                z_t_c_s_dummy = (zd_nominal, zd_ext, zd_settle[:, -1])

                all_zo.append(z_t_c_s)
                all_zd.append(z_t_c_s_dummy)


                delay_trial_ic[d, tr, ic_index] = ' '.join([str(delay), str(trial), str(ic_index)])

        all_error_out[d, : , :, :] = np.array(all_zo)
        all_error_dummy[d, :, :, :] = np.array(all_zd)


    return delay_trial_ic, all_error_out, all_error_dummy


def distraction_robustness(params, test_params, digits_rep, labels):

    '''
    adding noise to the first stimulus
    return all_fails --> for all trials in test_params['n_trials'] returns a 2D list of size = n_trials X 16
    16: there are 4 distinct trials ( 00, 01, 10, 11) and 4 distinct initial condition (i.e. trial conclusions), so overall there 
    are 4*4=16 possible cases
    '''
    net_prs = params['network']
    task_prs = params['task']
    model_prs = params['model']
    N = params['network']['N']
    wo, wd = model_prs['wo'], model_prs['wd']
    wf, wfd, wi = model_prs['wf'], model_prs['wfd'], model_prs['wi']
    J = model_prs['J']
    dt, tau, g = net_prs['dt'], net_prs['tau'], net_prs['g']
    fw_steps = int(1 * task_prs['t_trial'] / net_prs['dt'])
    seed = test_params['seed']
    trials = task_prs['keep_perms']
    fail = []

    for n in range(test_params['n_trial']):

        seed += 2
        for tr_index, trial in enumerate(trials):

            task = sum_task_experiment(task_prs['n_digits'], 0, 0, 1, task_prs['time_intervals'],
                                       net_prs['dt'], task_prs['output_encoding'], [trial], digits_rep, labels,
                                       seed)

            exp_mat, target_mat, dummy_mat, i_digits, output_digits = task.experiment()
            noisy_exp_mat = add_noise_to_stim(exp_mat, params, test_params, seed)
            zo_target = target_mat[:, -1]


            for ic_index in range(len(trials)):

                x = test_params['ICs']['x'][ic_index, :]
                r = np.tanh(x)
                z = np.matmul(wo.T, r)
                zd = np.matmul(wd.T, r)


                x_mat, r_mat = np.zeros([net_prs['N'], fw_steps]), np.zeros([net_prs['N'], fw_steps])

                for i in range(fw_steps):
                    x_mat[:, i] = x.reshape(-1)
                    r_mat[:, i] = r.reshape(-1)

                    dx = -x + g * np.matmul(J, r) + np.matmul(wf, z) + np.matmul(wi, noisy_exp_mat[:, i].reshape(
                        [net_prs['d_input'], 1])) + np.matmul(wfd, zd)

                    x = x + (dx * dt) / tau

                    r = np.tanh(x)
                    z = np.matmul(wo.T, r)
                    zd = np.matmul(wd.T, r)


                z_noisy = np.matmul(wo.T, r_mat[:,-1])
                initial_x = x_mat[:, -1]
                _, _, z_mat, zd_mat = forward_simulate(params, initial_x, test_params['n_fw_end']*fw_steps)
                zo_settle = z_mat[:, -1]
                evaluate_failure_type(zo_target, z_noisy, zo_settle, test_params, fail)

    return fail


def distraction_fail_ratio(params, test_params, digits_rep, labels):
    # evaluate how many trial fails out of all ics* trials * ph_params['n_trial']  as a function different noise levels
    # eg for 01 trials: 4*4*n_trial
    all_ratio = []

    for i in range(len(test_params['noise_sigma'])):

        test_params['stim_noise_sigma'] = test_params['noise_sigma'][i]
        fail = distraction_robustness(params, test_params, digits_rep, labels)
        fail_np = np.array(fail)
        num_fail = len(np.where(fail_np == 'True')[0])
        all_ratio.append(num_fail/(test_params['n_trial'] * (len(params['task']['keep_perms']) **2)))

    return all_ratio


def distraction_continous_error(params, test_params, digits_rep, labels):

    '''
    adding noise to the first stimulus
    return (z_target, z_noisy, z_settle) saved in all_noise_z: [n x 16 x 3]
    '''
    net_prs = params['network']
    task_prs = params['task']
    model_prs = params['model']
    N = params['network']['N']
    wo, wd = model_prs['wo'], model_prs['wd']
    wf, wfd, wi = model_prs['wf'], model_prs['wfd'], model_prs['wi']
    J = model_prs['J']
    dt, tau, g = net_prs['dt'], net_prs['tau'], net_prs['g']
    fw_steps = int(1 * task_prs['t_trial'] / net_prs['dt'])
    seed = test_params['seed']
    trials = task_prs['keep_perms']
    all_noise_z = np.zeros([test_params['n_trial'],len(trials)**2,3 ])

    for n in range(test_params['n_trial']):

        all_z = []
        seed += 2
        for tr_index, trial in enumerate(trials):

            task = sum_task_experiment(task_prs['n_digits'], 0, 0, 1, task_prs['time_intervals'],
                                       net_prs['dt'], task_prs['output_encoding'], [trial], digits_rep, labels,
                                       seed)

            exp_mat, target_mat, dummy_mat, i_digits, output_digits = task.experiment()
            noisy_exp_mat = add_noise_to_stim(exp_mat, params, test_params, seed)
            zo_target = target_mat[:, -1]


            for ic_index in range(len(trials)):

                x = test_params['ICs']['x'][ic_index, :]
                r = np.tanh(x)
                z = np.matmul(wo.T, r)
                zd = np.matmul(wd.T, r)


                x_mat, r_mat = np.zeros([net_prs['N'], fw_steps]), np.zeros([net_prs['N'], fw_steps])

                for i in range(fw_steps):
                    x_mat[:, i] = x.reshape(-1)
                    r_mat[:, i] = r.reshape(-1)

                    dx = -x + g * np.matmul(J, r) + np.matmul(wf, z) + np.matmul(wi, noisy_exp_mat[:, i].reshape(
                        [net_prs['d_input'], 1])) + np.matmul(wfd, zd)

                    x = x + (dx * dt) / tau

                    r = np.tanh(x)
                    z = np.matmul(wo.T, r)
                    zd = np.matmul(wd.T, r)


                z_noisy = np.matmul(wo.T, r_mat[:,-1])
                initial_x = x_mat[:, -1]
                _, _, z_mat, zd_mat = forward_simulate(params, initial_x, test_params['n_fw_end']*fw_steps)
                zo_settle = z_mat[:, -1]
                z_t_n_s = (zo_target, z_noisy, zo_settle)
                all_z.append(z_t_n_s)

        all_noise_z[n,:,:] = np.array(all_z).squeeze()


    return all_noise_z


def get_allnoise_errors(params, test_params, digits_rep, labels):
    # evaluate how many trial fails out of all ics* trials * ph_params['n_trial']  as a function different noise levels
    # eg for 01 trials: 4*4*n_trial
    all_noise_error = np.zeros([len(test_params['noise_sigma']), test_params['n_trial'], len(params['task']['keep_perms'])**2, 3])

    for i in range(len(test_params['noise_sigma'])):

        test_params['stim_noise_sigma'] = test_params['noise_sigma'][i]
        all_noise_z = distraction_continous_error(params, test_params, digits_rep, labels)  # n x 16 x 3
        all_noise_error[i, :,:,:] = all_noise_z

    return all_noise_error


def interruption_robustness(params, test_params, digits_rep, labels):
    # same as distraction but for off neurons
    net_prs = params['network']
    task_prs = params['task']
    model_prs = params['model']
    wo, wd = model_prs['wo'], model_prs['wd']
    wf, wfd, wi = model_prs['wf'], model_prs['wfd'], model_prs['wi']
    J = model_prs['J']
    dt, tau, g = net_prs['dt'], net_prs['tau'], net_prs['g']
    fw_steps = int(1 * task_prs['t_trial'] / net_prs['dt'])
    stim1 = int(task_prs['time_intervals']['stim_on'] / net_prs['dt'])
    seed = test_params['seed']
    trials = task_prs['keep_perms']

    wi_corr = wi
    wi_corr[:test_params['n_off']] = 0.
    fail = []

    for n in range(test_params['n_trial']):

        seed += 1

        for trial in trials:

            task = sum_task_experiment(task_prs['n_digits'], 0, 0, 1, task_prs['time_intervals'],
                                       net_prs['dt'], task_prs['output_encoding'], [trial], digits_rep, labels,
                                       seed)

            exp_mat, target_mat, dummy_mat, i_digits, output_digits = task.experiment()
            zo_target = target_mat[:, -1]

            for ic_index in range(len(trials)):
                x = test_params['ICs']['x'][ic_index, :]
                r = np.tanh(x)
                z = np.matmul(wo.T, r)
                zd = np.matmul(wd.T, r)

                x_mat, r_mat = np.zeros([net_prs['N'], fw_steps]), np.zeros([net_prs['N'], fw_steps])

                for i in range(fw_steps):
                    x_mat[:, i] = x.reshape(-1)
                    r_mat[:, i] = r.reshape(-1)

                    if i < stim1:

                        dx = -x + g * np.matmul(J, r) + np.matmul(wf, z) + np.matmul(wi_corr, exp_mat[:, i].reshape(
                            [net_prs['d_input'], 1])) + np.matmul(wfd, zd)

                        x = x + (dx * dt) / tau
                    else:
                        dx = -x + g * np.matmul(J, r) + np.matmul(wf, z) + np.matmul(wi, exp_mat[:, i].reshape(
                            [net_prs['d_input'], 1])) + np.matmul(wfd, zd)

                        x = x + (dx * dt) / tau


                    r = np.tanh(x)
                    z = np.matmul(wo.T, r)
                    zd = np.matmul(wd.T, r)

                z_corr = np.matmul(wo.T, r_mat[:, -1])
                initial_x = x_mat[:, -1]
                r_mat, x_mat, z_mat, zd_mat = forward_simulate(params, initial_x, test_params['n_fw_end'] * fw_steps)
                zo_settle = z_mat[:, -1]
                evaluate_failure_type(zo_target, z_corr, zo_settle, test_params, fail)

    return fail


def interruption_fail_ratio(params, test_params, digits_rep, labels):
    all_ratio = []

    for i in range(len(test_params['n_off_list'])):
        test_params['n_off'] = test_params['n_off_list'][i]
        fail = interruption_robustness(params, test_params, digits_rep, labels)
        fail_np = np.array(fail)
        num_fail = len(np.where(fail_np == 'True')[0])
        all_ratio.append(num_fail / (test_params['n_trial'] * (len(params['task']['keep_perms']) ** 2)))

    return all_ratio


def interruption_continous_error(params, test_params, digits_rep, labels):
    # same as distraction but for off neurons
    net_prs = params['network']
    task_prs = params['task']
    model_prs = params['model']
    wo, wd = model_prs['wo'], model_prs['wd']
    wf, wfd, wi = model_prs['wf'], model_prs['wfd'], model_prs['wi']
    J = model_prs['J']
    dt, tau, g = net_prs['dt'], net_prs['tau'], net_prs['g']
    fw_steps = int(1 * task_prs['t_trial'] / net_prs['dt'])
    stim1 = int(task_prs['time_intervals']['stim_on'] / net_prs['dt'])
    seed = test_params['seed']
    trials = task_prs['keep_perms']

    wi_corr = wi
    wi_corr[:test_params['n_off']] = 0.
    all_off_z = np.zeros([test_params['n_trial'], len(trials) ** 2, 3])

    for n in range(test_params['n_trial']):

        all_z = []
        seed += 1

        for trial in trials:

            task = sum_task_experiment(task_prs['n_digits'], 0, 0, 1, task_prs['time_intervals'],
                                       net_prs['dt'], task_prs['output_encoding'], [trial], digits_rep, labels,
                                       seed)

            exp_mat, target_mat, dummy_mat, i_digits, output_digits = task.experiment()
            zo_target = target_mat[:, -1]

            for ic_index in range(len(trials)):
                x = test_params['ICs']['x'][ic_index, :]
                r = np.tanh(x)
                z = np.matmul(wo.T, r)
                zd = np.matmul(wd.T, r)

                x_mat, r_mat = np.zeros([net_prs['N'], fw_steps]), np.zeros([net_prs['N'], fw_steps])

                for i in range(fw_steps):
                    x_mat[:, i] = x.reshape(-1)
                    r_mat[:, i] = r.reshape(-1)

                    if i < stim1:

                        dx = -x + g * np.matmul(J, r) + np.matmul(wf, z) + np.matmul(wi_corr, exp_mat[:, i].reshape(
                            [net_prs['d_input'], 1])) + np.matmul(wfd, zd)

                        x = x + (dx * dt) / tau
                    else:

                        dx = -x + g * np.matmul(J, r) + np.matmul(wf, z) + np.matmul(wi, exp_mat[:, i].reshape(
                            [net_prs['d_input'], 1])) + np.matmul(wfd, zd)

                        x = x + (dx * dt) / tau

                    r = np.tanh(x)
                    z = np.matmul(wo.T, r)
                    zd = np.matmul(wd.T, r)

                z_corr = np.matmul(wo.T, r_mat[:, -1])
                initial_x = x_mat[:, -1]
                r_mat, x_mat, z_mat, zd_mat = forward_simulate(params, initial_x, test_params['n_fw_end'] * fw_steps)
                zo_settle = z_mat[:, -1]
                z_t_c_s = (zo_target, z_corr, zo_settle)
                all_z.append(z_t_c_s)

        all_off_z[n,:,:] = np.array(all_z).squeeze()


    return all_off_z


def get_alloff_errors(params, test_params, digits_rep, labels):

    all_off_error = np.zeros([len(test_params['n_off_list']), test_params['n_trial'], len(params['task']['keep_perms'])**2, 3])

    for i in range(len(test_params['n_off_list'])):

        test_params['n_off'] = test_params['n_off_list'][i]
        all_z = interruption_continous_error(params, test_params, digits_rep, labels)
        all_off_error[i,:,:,:] = all_z

    return all_off_error


def save_data_variable_size(*vars1, name=None, prefix='success', dir=None):
    file_name = prefix + '_' + name
    with open(dir + file_name, 'wb') as f:
        pickle.dump((vars1), f, protocol=-1)



def extend_nominal_delays_save_r(params, test_params, digits_rep, labels):
    # extend first delay, and evaluate the failure type (does it even fail? asymptotic or fixed_time)
    # returns all_fail : shape= len(all_delayes) X 16

    trials = params['task']['keep_perms']

    R_mat = np.zeros([len(test_params['extend_delays']), len(trials)**2, params['network']['N']])
    R_mat_ext = np.zeros([len(test_params['extend_delays']), len(trials) ** 2, params['network']['N']])

    for d, delay in enumerate(test_params['extend_delays']):

        R = np.zeros([params['network']['N'], 1])
        R_ext = np.zeros([ params['network']['N'], 1])

        for tr, trial in enumerate(trials):

            for ic_index in range(len(trials)):
                r_mat, _, r_extend, _ = one_trial(params, test_params, digits_rep, labels,
                                               trial, ic_index, extend=True, delay_wm=delay)


                R = np.concatenate((R, r_mat[:, -1][:,np.newaxis]), axis=1)
                R_ext = np.concatenate((R_ext, r_extend[:, -1][:,np.newaxis]), axis=1)


        R_mat[d, :, :] = R[:, 1:].T
        R_mat_ext[d, :, : ] = R_ext[:, 1:].T

    return R_mat, R_mat_ext

test_params = set_test_params(x_ICs, r_ICs)
labels, digits_rep = get_digits_reps()

R_mat, R_mat_ext = extend_nominal_delays_save_r(params, test_params, digits_rep, labels)
#delay_trial_ic, all_error_out, all_error_dummy = extend_nominal_delays_error(params, test_params, digits_rep, labels)
all_noise_error = get_allnoise_errors(params, test_params, digits_rep, labels)
# all_off_error = get_alloff_errors(params, test_params, digits_rep, labels)
# all_fail, delay_trial_ic = extend_nominal_delays(params, test_params, digits_rep, labels)
#all_dst = distraction_fail_ratio(params, test_params, digits_rep, labels)
# all_intr = interruption_fail_ratio(params, test_params, digits_rep, labels)
all_off_error = None
# all_fail, dtic = None ,None
# all_intr = None
save_data_variable_size(params, all_noise_error, all_off_error, name=file_name, prefix='success', dir=outdir)
#save_data_variable_size(params, R_mat, R_mat_ext, name=file_name, prefix='r_success', dir=outdir)

#save_data_variable_size(params, all_fail, dtic, all_dst, all_intr, name=file_name, prefix='1success', dir=outdir)
#save_data_variable_size(params, delay_trial_ic, all_error_out, all_error_dummy, name=file_name, prefix='success', dir=outdir)

print('tests done and saved')
























