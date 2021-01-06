"""
main file to train/test RNN and perform posthoc tests on WashU cluster

written in Python 3.8.3
@ Elham
"""
import argparse
import json
import sys
from SPM_task import *
from train_force import *
from posthoc_tests import *
dir = '/scratch/elham/results3500c/'

parser = argparse.ArgumentParser()
parser.add_argument('-d', type=json.loads)
args = parser.parse_args()
kwargs= args.d


def set_all_parameters( g, pg, fb_var, input_var,  n_train, encoding, seed, init_dist, activation='tanh', isFORCE = False):
    params = dict()

    net_params = dict()
    net_params['d_input'] = 2
    net_params['d_output'] = 1
    net_params['tau'] = 1
    net_params['dt'] = 0.1
    net_params['g'] = g
    net_params['pg'] = pg
    net_params['N'] = 1000
    net_params['fb_var'] = fb_var
    net_params['input_var'] = input_var
    params['network'] = net_params

    task_params = dict()
    t_intervals = dict()
    t_intervals['fixate_on'], t_intervals['fixate_off'] = 0, 0
    t_intervals['cue_on'], t_intervals['cue_off'] = 0, 0
    t_intervals['stim_on'], t_intervals['stim_off'] = 10, 5
    t_intervals['delay_task'] = 0
    t_intervals['response'] = 5
    task_params['time_intervals'] = t_intervals
    task_params['t_trial'] = sum(t_intervals.values()) + t_intervals['stim_on'] + t_intervals['stim_off']
    task_params['output_encoding'] = encoding  # how 0, 1, 2 are encoded
    task_params['keep_perms'] = [(0, 0), (0, 1), (1, 0), (1, 1)]
    task_params['n_digits'] = 9
    params['task'] = task_params

    train_params = dict()
    train_params['update_step'] = 2  # update steps of FORCE
    train_params['alpha_w'] = 1.
    train_params['alpha_d'] = 1.
    train_params['n_train'] = n_train  # training steps
    train_params['n_train_ext'] = 0
    train_params['n_test'] = 20   # test steps
    train_params['init_dist'] = init_dist
    train_params['activation'] = activation
    train_params['FORCE'] = isFORCE
    train_params['epsilon'] = [0.005, 0.01, 0.05, 0.1]
    params['train'] = train_params

    other_params = dict()
    other_params['name'] = '_'.join(['{}'.format(val) if type(val) != list
                                     else '{}'.format(''.join([str(s) for s in val])) for k, val in kwargs.items()])
    print('name is = ',other_params['name']  )
    #str(task_params['output_encoding']) + '_g' + str(net_params['g']) + '_' +
    #  str(train_params['n_train']+ train_params['n_train_ext'])+ 'Gauss_S' + 'FORCE'
    other_params['n_plot'] = 10
    other_params['seed'] = seed  #default is 0
    params['msc'] = other_params

    return params


def get_digits_reps():

    with open('allDigCNNMNIST', 'rb') as f:
        z_mean, z_log_var, z_sample, x_test, y_test = pickle.load(f)

    y_test, x_test = np.array(y_test), x_test.reshape([x_test.shape[0], 28, 28])

    return y_test, z_sample


params = set_all_parameters(**kwargs)

labels, digits_rep = get_digits_reps()
task_prs = params['task']
train_prs = params['train']
net_prs = params['network']
msc_prs = params['msc']
task = sum_task_experiment(task_prs['n_digits'], train_prs['n_train'], train_prs['n_train_ext'], train_prs['n_test'], task_prs['time_intervals'],
                           net_prs['dt'], task_prs['output_encoding'], task_prs['keep_perms'] , digits_rep, labels, msc_prs['seed'])

exp_mat, target_mat, dummy_mat, input_digits, output_digits = task.experiment()

if not train_prs['FORCE']:
    print('FORCE Reinforce IS RUNNING\n')
    x_train, params = train(params, exp_mat, target_mat, dummy_mat, input_digits, dist=train_prs['init_dist'])


elif train_prs['FORCE']:
    print('FORCE IS RUNNING\n')
    x_train, params = train_FORCE(params, exp_mat, target_mat, dummy_mat, input_digits, dist=train_prs['init_dist'])

x_ICs, r_ICs, internal_x = test(params, x_train, exp_mat, target_mat, dummy_mat, input_digits)

error_ratio = error_rate(params, x_ICs, digits_rep, labels)


save_data_variable_size(params, x_ICs, r_ICs, error_ratio, name=params['msc']['name'], prefix='train', dir=dir)

###############################################################################
#                             Post-hoc Tests                                  #
###############################################################################

ph_params = set_posthoc_params(x_ICs, r_ICs)

#r_fw, _ = r_for_pca(params, ph_params, digits_rep, labels)

trajectories, unique_z_mean, unique_zd_mean, attractor = attractor_type(params, ph_params, digits_rep, labels)

_, _, _, attractor_nozero = attractor_type_nozero(params, ph_params, digits_rep, labels)


all_F_norms = asymp_trial_end(params, ph_params)

#all_delay_traj, all_z, all_zd = vf_delays(params, ph_params, digits_rep, labels)

save_data_variable_size(ph_params,  unique_z_mean, unique_zd_mean, attractor, attractor_nozero,
                        all_F_norms,  name=params['msc']['name'], prefix='fp_test', dir=dir)

#trials_failed = extend_nominal_delays(params, ph_params, digits_rep, labels)

#dsct_failed_ratio = distraction_fail_ratio(params, ph_params, digits_rep, labels)

#intrp_failed_ratio = interruption_fail_ratio(params, ph_params, digits_rep, labels)

sat_ratio_delay, sat_ratio_resp = saturation_percentage(params, ph_params, digits_rep, labels)

trials_failed, dsct_failed_ratio, intrp_failed_ratio = 0, 0 , 0

save_data_variable_size(trials_failed, dsct_failed_ratio, intrp_failed_ratio,
                        sat_ratio_delay, sat_ratio_resp, name=params['msc']['name'], prefix='ext_test', dir=dir)


print('DONE')





















