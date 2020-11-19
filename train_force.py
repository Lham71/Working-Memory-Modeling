"""
Functions for train RNN using modified version of FORCE (i.e. temporally restricted error kernel).
Functions for test, train RNN using original FORCE, save data, training/test performance plots, evaluate error rate
written in Python 3.8.3
@ Elham
"""

print('train_force is executing\n')
import numpy as np
from scipy import sparse
from matplotlib import pyplot as plt
from drawnow import *
from SPM_task import *
import time
import pickle



def initialize_net(params, dist="Gauss"):
    '''
    initialize network states and initial params
    Args:
    params: Dictionary containing all parameters
    dist: Distribution for initialization of weights and states -- can be either 'Gauss' or 'Uniform'
    '''
    net_prs = params['network']
    train_prs = params['train']
    msc_prs = params['msc']
    N = net_prs['N']
    rng = np.random.RandomState(msc_prs['seed'])
    Pw = np.eye(N)/train_prs['alpha_w']  #inverse correlation matrix
    Pd = np.eye(N)/train_prs['alpha_d']

    std = 1/np.sqrt(net_prs['pg'] * N)
    J = std * sparse.random(N, N, density=net_prs['pg'],
                            random_state=msc_prs['seed'], data_rvs=rng.randn).toarray()  #connectivity matrix


    if dist == 'Gauss':

        x = 0.1 * rng.randn(N, 1)
        wf = (1. * rng.randn(N, net_prs['d_output'])) / net_prs['fb_var']
        wi = (1. * rng.randn(N, net_prs['d_input'])) / net_prs['input_var']
        wfd = (1. * rng.randn(N, net_prs['d_input'])) / net_prs['fb_var']



    elif dist == 'Uniform':
        print('initialization is uniform')
        x = 0.1 * (2 * rng.rand(N, 1) -1)
        wf = (2 * rng.rand(N, net_prs['d_output']) - 1) / net_prs['fb_var']
        wi = (2 * rng.rand(N, net_prs['d_input']) - 1) / net_prs['input_var']
        wfd = (2 * rng.rand(N, net_prs['d_input']) - 1) / net_prs['fb_var']


    wo = np.zeros([N, net_prs['d_output']])
    #wfd = wfd * 0.
    wd = np.zeros([N, net_prs['d_input']])

    return Pw, Pd, J, x, wf, wo, wfd, wd, wi


def zero_fat_mats(params, is_train=True):

    '''
    initialize zero matrix
    '''

    net_prs = params['network']
    train_prs = params['train']
    task_prs = params['task']
    if is_train:
        total_size = train_prs['n_train'] + train_prs['n_train_ext']
    elif not is_train:
        total_size = train_prs['n_test']

    total_steps = int(total_size * task_prs['t_trial'] / net_prs['dt'])
    z_mat = np.zeros(total_steps)
    zd_mat = np.zeros([net_prs['d_input'], total_steps])
    x_mat = np.zeros([net_prs['N'], total_steps])
    r_mat = np.zeros([net_prs['N'], total_steps])
    wo_dot = np.zeros([total_steps, net_prs['d_output']])
    wd_dot = np.zeros([total_steps, net_prs['d_input']])

    return z_mat, zd_mat, x_mat, r_mat, wo_dot, wd_dot


def train(params, exp_mat, target_mat, dummy_mat, input_digits, dist='Gauss'):

    """
    Main function to implement training using modified FORCE algorithm
    exp_mat: sequence of trials (d_input X trial_len*trial_size)
    target_mat: target signal for training network output zo (d_output X trial_len*trial_size)
    dummy_mat: target signal for training dummy outputs (memory encoding) zd (d_input X trial_len*trial_size)
    input_digits: digits in trials
    Return: x (final state) and params (updated with trained weights)
    """

    tic = time.time()
    net_prs = params['network']
    train_prs = params['train']
    task_prs = params['task']
    msc_prs = params['msc']
    dt, tau, g = net_prs['dt'], net_prs['tau'], net_prs['g']
    update_step = train_prs['update_step']
    train_steps = int((train_prs['n_train'] + train_prs['n_train_ext'])* task_prs['t_trial'] / net_prs['dt'])
    time_steps = np.arange(0, train_steps, 1)

    # initialization
    Pw, Pd, J, x, wf, wo, wfd, wd, wi = initialize_net(params, dist=dist)
    r = np.tanh(x)
    z = np.matmul(wo.T, r)
    zd = np.matmul(wd.T, r)

    z_mat, zd_mat, x_mat, r_mat, wo_dot, wd_dot = zero_fat_mats(params, is_train=True)

    trial = 0
    plt_c = 0
    s = 0
    si = 0

    # start training
    for i in range(train_steps):
        z_mat[i] = z
        zd_mat[:, i] = zd.reshape(-1)
        x_mat[:, i] = x.reshape(-1)
        r_mat[:, i] = r.reshape(-1)

        dx = -x + g * np.matmul(J, r) + np.matmul(wf, z) + \
             np.matmul(wi, exp_mat[:, i].reshape([net_prs['d_input'], 1])) + np.matmul(wfd, zd)


        x = x + (dx * dt) / tau
        # if i%1000 == 0:
        #     plt.hist(x)
        #     plt.show()
        r = np.tanh(x)
        z = np.matmul(wo.T, r)

        zd = np.matmul(wd.T, r)

        if np.all(dummy_mat[:, i] != 0.):

            if i % update_step == 0 and i >= update_step :
                #print('I AM BUG')

                Pdr = np.matmul(Pd, r)
                num_pd = np.outer(Pdr, np.matmul(r.T, Pd))
                denum_pd = 1 + np.matmul(r.T, Pdr)
                Pd -= num_pd / denum_pd

                target_d = np.reshape(dummy_mat[:, i], [net_prs['d_input'], 1])
                ed_ = zd - target_d

                Delta_wd = np.outer(Pdr, ed_) / denum_pd
                wd -= Delta_wd

                wd_dot[i, :] = np.linalg.norm(Delta_wd / (update_step * dt), axis=0, keepdims=True)


        if np.any(target_mat[:, i] != 0.):

            if i % update_step == 0 and i >= update_step:
                Pr = np.matmul(Pw, r)
                num_pw = np.outer(Pr, np.matmul(r.T, Pw))
                denum_pw = 1 + np.matmul(r.T, Pr)
                Pw -= num_pw / denum_pw

                target = np.reshape(target_mat[:, i], [net_prs['d_output'], 1])
                e_ = z - target

                # Delta_w = onp.outer(Pr, e_)
                Delta_w = np.outer(Pr, e_) / denum_pw
                wo -= Delta_w

                wo_dot[i] = np.linalg.norm(Delta_w / (update_step * dt))

        # plot
        def draw_output():

            # plt.subplot(211)
            plt.title(n1n2)
            plt.plot(time_steps[s:i], target_mat[:, s:i].T, c='gray')
            plt.plot(time_steps[s:i], z_mat[s:i], c='c')

            # plt.subplot(212)
            # plt.plot(time_steps[s:i], wo_dot[s:i])


        def draw_input():

            ax2 = plt.subplot(211)
            ax2.set_title(n1n2)

            ax2.plot(time_steps[s:i], exp_mat[0, s:i], c='b')
            ax2.plot(time_steps[s:i], exp_mat[1, s:i] - 3, c='k')

            ax3 = plt.subplot(212)
            ax3.plot(time_steps[s:i], dummy_mat[0, s:i], 'c--')
            ax3.plot(time_steps[s:i], dummy_mat[1, s:i] - 10, 'g--')
            ax3.plot(time_steps[s:i], zd_mat[0, s:i], 'c')
            ax3.plot(time_steps[s:i], zd_mat[1, s:i] - 10, 'g')

            # ax4 = plt.subplot(313)
            # ax4.plot(time_steps[s:i], wd_dot[s:i, 0] + 2)
            # ax4.plot(time_steps[s:i], wd_dot[s:i, 1])

        if i % int(msc_prs['n_plot'] * task_prs['t_trial'] / net_prs['dt'])  == 0 and i != 0:


            n1n2 = str(input_digits[plt_c: plt_c + msc_prs['n_plot']])
            # plt.figure(num=1, figsize=(14, 7))
            # drawnow(draw_output)
            # plt.pause(2)
            # plt.figure(num=2, figsize=(14, 7))
            # drawnow(draw_input)

            s = i
            plt_c += msc_prs['n_plot']

        #plot gradients
        # if i % 350 * 200 == 0 and i != 0 and i>350 * (train_prs['n_train']-10) :
        #     wo_dot = wo_dot.squeeze()
        #     wo_grad = wo_dot[:i][np.argwhere(wo_dot[:i])]
        #     #print('wo_grad', wo_grad.shape)
        #     plt.figure(figsize=(14,7), num=1)
        #     plt.title(params['msc']['name'])
        #     plt.plot(wo_grad, c='gray')
        #     plt.axhline(y=0.01, linestyle='--')
        #     plt.pause(0.01)
        #     plt.show(block=False)
        #     si = i

    toc = time.time()
    print('\n', 'train time = ' , (toc-tic)/60)
    print('read out norm = ', np.linalg.norm(wo))
    print('dummy norm = ', np.linalg.norm(wd, axis=0, keepdims=True))

    model_params = {'J':J, 'g':g, 'wf':wf, 'wo':wo, 'wfd':wfd, 'wd':wd, 'wi':wi}
    params['model'] = model_params
    task_prs['counter'] = i
    return x, params


def test(params, x_train, exp_mat, target_mat, dummy_mat, input_digits):
    """
    Function to visualize if network has learned the task and get trial conclusion as initial condition of next trial

    x_train: final state of network at training phase
    Return: initial state and activity
    """
    net_prs = params['network']
    train_prs = params['train']
    task_prs = params['task']
    msc_prs = params['msc']
    model_prs = params['model']
    wo, wd = model_prs['wo'], model_prs['wd']
    wf, wfd, wi = model_prs['wf'], model_prs['wfd'], model_prs['wi']
    J = model_prs['J']
    dt, tau, g = net_prs['dt'], net_prs['tau'], net_prs['g']
    test_steps = int(train_prs['n_test'] * task_prs['t_trial'] / net_prs['dt'])
    time_steps = np.arange(0, test_steps, 1)
    counter = task_prs['counter']
    exp_mat = exp_mat[:, counter+1:]
    target_mat = target_mat[:, counter+1:]
    dummy_mat = dummy_mat[:, counter+1:]
    test_digits = input_digits[train_prs['n_train']+ train_prs['n_train_ext']:]

    i00, i01, i10, i11 = 0, 0, 0, 0
    i02, i20, i22, i12, i21 = 0, 0, 0, 0, 0

    x = x_train
    r = np.tanh(x)
    z = np.matmul(wo.T, r)
    zd = np.matmul(wd.T, r)

    z_mat, zd_mat, x_mat, r_mat, wo_dot, wd_dot = zero_fat_mats(params, is_train=False)
    trial = 0
    s = 0
    plt_c = 0

    for i in range(test_steps):

        z_mat[i] = z
        zd_mat[:, i] = zd.reshape(-1)
        x_mat[:, i] = x.reshape(-1)
        r_mat[:, i] = r.reshape(-1)

        dx = -x + g * np.matmul(J, r) + np.matmul(wf, z) + \
             np.matmul(wi, exp_mat[:, i].reshape([net_prs['d_input'], 1])) + np.matmul(wfd, zd)


        x = x + (dx * dt) / tau
        r = np.tanh(x)
        z = np.matmul(wo.T, r)

        zd = np.matmul(wd.T, r)

        # save initial condition
        if i % int((task_prs['t_trial'])/ dt ) == 0 and i != 0:


            if test_digits[trial][1] == (0,0) and i00 == 0:

                r00 = r_mat[:, i-1][:, np.newaxis]
                x00 = x_mat[:, i-1][:, np.newaxis]
                i00 = 1

            elif test_digits[trial][1] == (0,1) and i01 == 0:

                r01 = r_mat[:, i-1][:, np.newaxis]
                x01 = x_mat[:, i-1][:, np.newaxis]
                i01 = 1

            elif test_digits[trial][1] == (1,0) and i10 == 0:

                r10 = r_mat[:, i-1][:, np.newaxis]
                x10 = x_mat[:, i-1][:, np.newaxis]
                i10 = 1

            elif test_digits[trial][1] == (1,1) and i11 == 0:

                r11 = r_mat[:, i-1][:, np.newaxis]
                x11 = x_mat[:, i-1][:, np.newaxis]
                i11 = 1

            elif test_digits[trial][1] == (0, 2) and i02 == 0:

                r02 = r_mat[:, i - 1][:, np.newaxis]
                x02 = x_mat[:, i - 1][:, np.newaxis]
                i02 = 1

            elif test_digits[trial][1] == (2, 0) and i20 == 0:

                r20 = r_mat[:, i - 1][:, np.newaxis]
                x20 = x_mat[:, i - 1][:, np.newaxis]
                i20 = 1

            elif test_digits[trial][1] == (2, 2) and i22 == 0:

                r22 = r_mat[:, i - 1][:, np.newaxis]
                x22 = x_mat[:, i - 1][:, np.newaxis]
                i22 = 1

            elif test_digits[trial][1] == (1, 2) and i12 == 0:

                r12 = r_mat[:, i - 1][:, np.newaxis]
                x12 = x_mat[:, i - 1][:, np.newaxis]
                i12 = 1

            elif test_digits[trial][1] == (2, 1) and i21 == 0:

                r21 = r_mat[:, i - 1][:, np.newaxis]
                x21 = x_mat[:, i - 1][:, np.newaxis]
                i21 = 1

            trial += 1


        # plot
        def draw_output():


            plt.subplot(211)
            plt.title(n1n2 + '\n' + params['msc']['name'])
            plt.plot(time_steps[ s:i], target_mat[:, s:i].T, c='gray')
            plt.plot(time_steps[ s:i], z_mat[s:i], c='c')

            plt.subplot(212)
            plt.plot(time_steps[ s:i], wo_dot[s:i])

        def draw_input():

            ax2 = plt.subplot(311)
            ax2.set_title(n1n2)

            ax2.plot(time_steps[s:i], exp_mat[0, s:i], c='b')
            ax2.plot(time_steps[s:i], exp_mat[1, s:i] - 3, c='k')

            ax3 = plt.subplot(312)
            ax3.plot(time_steps[s:i], dummy_mat[0, s:i], 'c--')
            ax3.plot(time_steps[s:i], dummy_mat[1, s:i] - 10, 'g--')
            ax3.plot(time_steps[s:i], zd_mat[0, s:i], 'c')
            ax3.plot(time_steps[s:i], zd_mat[1, s:i] - 10, 'g')

            ax4 = plt.subplot(313)
            ax4.plot(time_steps[s:i], wd_dot[s:i, 0] + 2)
            ax4.plot(time_steps[s:i], wd_dot[s:i, 1])


        if i % int(msc_prs['n_plot'] * task_prs['t_trial'] / net_prs['dt'])  == 0 and i != 0:

            n1n2 = str(test_digits[plt_c: plt_c + msc_prs['n_plot']])
            # plt.figure(num=1, figsize=(14,7))
            #
            # drawnow(draw_output)
            #
            # # plt.figure(num=2, figsize=(14, 7))
            # # drawnow(draw_input)
            # plt.pause(5)
            #
            # s = i
            # plt_c += msc_prs['n_plot']

    # x_ICs = np.array([x00, x01, x10, x11, x02, x20, x12, x21, x22])
    # r_ICs = np.array([r00, r01, r10, r11, r02, r20, r12, r21, r22])
    x_ICs = np.array([x00, x01, x10, x11])
    r_ICs = np.array([r00, r01, r10, r11])

    return  x_ICs, r_ICs, x_mat


def save_data(name, params, x_ICs, r_ICs):
    '''
    function to save data and both initial and trained parameters for 3D plots and PCA
    it should be used inside this script -- to save from train() and interrogate() functions
    '''
    dir = '/Users/elhamghazizadeh/Desktop/WM/MainResults/Main/'

    with open(dir + 'trainParams--' + name, 'wb') as f:
        pickle.dump([params, x_ICs, r_ICs], f)


def save_data_any(*vars1, name=None, prefix='train', dir = None):

    file_name = prefix + '_' + name
    with open(dir + file_name, 'wb') as f:
        pickle.dump((vars1), f, protocol=-1)


def save_data_variable_size(*vars1, name=None, prefix='train', dir=None):
    file_name = prefix + '_' + name
    with open(dir + file_name, 'wb') as f:
        pickle.dump((vars1), f, protocol=-1)


def train_FORCE(params, exp_mat, target_mat, dummy_mat, input_digits, dist='Gauss'):
    '''
    train using original FORCE with no modification
    '''

    tic = time.time()
    net_prs = params['network']
    train_prs = params['train']
    task_prs = params['task']
    msc_prs = params['msc']
    dt, tau, g = net_prs['dt'], net_prs['tau'], net_prs['g']
    update_step = train_prs['update_step']
    train_steps = int((train_prs['n_train']+ train_prs['n_train_ext'] )* task_prs['t_trial'] / net_prs['dt'])
    time_steps = np.arange(0, train_steps, 1)

    # initialization
    Pw, Pd, J, x, wf, wo, wfd, wd, wi = initialize_net(params, dist=dist)
    r = np.tanh(x)
    z = np.matmul(wo.T, r)
    zd = np.matmul(wd.T, r)

    z_mat, zd_mat, x_mat, r_mat, wo_dot, wd_dot = zero_fat_mats(params, is_train=True)

    trial = 0
    plt_c = 0
    s = 0

    # start training
    for i in range(train_steps):
        z_mat[i] = z
        zd_mat[:, i] = zd.reshape(-1)
        x_mat[:, i] = x.reshape(-1)
        r_mat[:, i] = r.reshape(-1)

        dx = -x + g * np.matmul(J, r) + np.matmul(wf, z) + \
             np.matmul(wi, exp_mat[:, i].reshape([net_prs['d_input'], 1])) + np.matmul(wfd, zd)


        x = x + (dx * dt) / tau
        # if i%1000 == 0:
        #     plt.hist(x)
        #     plt.show()
        r = np.tanh(x)
        z = np.matmul(wo.T, r)

        zd = np.matmul(wd.T, r)

        # update dummy
        if i % update_step == 0 and i >= update_step:
            #print('I AM BUG')

            Pdr = np.matmul(Pd, r)
            num_pd = np.outer(Pdr, np.matmul(r.T, Pd))
            denum_pd = 1 + np.matmul(r.T, Pdr)
            Pd -= num_pd / denum_pd

            target_d = np.reshape(dummy_mat[:, i], [net_prs['d_input'], 1])
            ed_ = zd - target_d

            Delta_wd = np.outer(Pdr, ed_) / denum_pd
            wd -= Delta_wd

            wd_dot[i, :] = np.linalg.norm(Delta_wd / (update_step * dt), axis=0, keepdims=True)


        # update readout
        if i % update_step == 0 and i >= update_step:
            Pr = np.matmul(Pw, r)
            num_pw = np.outer(Pr, np.matmul(r.T, Pw))
            denum_pw = 1 + np.matmul(r.T, Pr)
            Pw -= num_pw / denum_pw

            target = np.reshape(target_mat[:, i], [net_prs['d_output'], 1])
            e_ = z - target

            # Delta_w = onp.outer(Pr, e_)
            Delta_w = np.outer(Pr, e_) / denum_pw
            wo -= Delta_w

            wo_dot[i] = np.linalg.norm(Delta_w / (update_step * dt))

        # plot
        def draw_output():

            plt.subplot(211)
            #plt.title(n1n2)
            plt.plot(time_steps[s:i], target_mat[:, s:i].T, c='gray')
            plt.plot(time_steps[s:i], z_mat[s:i], c='c')

            plt.subplot(212)
            plt.plot(time_steps[s:i], wo_dot[s:i])

        def draw_input():

            ax2 = plt.subplot(311)
            #ax2.set_title(n1n2)

            ax2.plot(time_steps[s:i], exp_mat[0, s:i], c='b')
            ax2.plot(time_steps[s:i], exp_mat[1, s:i] - 3, c='k')

            ax3 = plt.subplot(312)
            ax3.plot(time_steps[s:i], dummy_mat[0, s:i], 'c--')
            ax3.plot(time_steps[s:i], dummy_mat[1, s:i] - 10, 'g--')
            ax3.plot(time_steps[s:i], zd_mat[0, s:i], 'c')
            ax3.plot(time_steps[s:i], zd_mat[1, s:i] - 10, 'g')

            ax4 = plt.subplot(313)
            ax4.plot(time_steps[s:i], wd_dot[s:i, 0] + 2)
            ax4.plot(time_steps[s:i], wd_dot[s:i, 1])

        # if i % int(msc_prs['n_plot'] * task_prs['t_trial'] / net_prs['dt'])  == 0 and i != 0:
        #
        #     # plt.figure(num=1, figsize=(14,7))
        #     n1n2 = str(input_digits[plt_c: plt_c + msc_prs['n_plot']])
        #     # drawnow(draw_output)
        #     # plt.pause(2)
        #     # plt.figure(num=2, figsize=(14, 7))
        #     # drawnow(draw_input)
        #
        #     s = i
        #     plt_c += msc_prs['n_plot']

            # plot gradients
        # if i % 350 * 200 == 0 and i != 0 and i > 350 * (train_prs['n_train'] - 10):
        #     wo_dot = wo_dot.squeeze()
        #     wo_grad = wo_dot[:i][np.argwhere(wo_dot[:i])]
            # print('wo_grad', wo_grad.shape)
            # plt.figure(figsize=(14, 7), num=1)
            # plt.title(params['msc']['name'])
            # plt.plot(wo_grad, c='gray')
            # plt.axhline(y=0.01, linestyle='--')
            # plt.pause(0.01)
            # plt.show(block=False)
            # si = i

    toc = time.time()
    print('\n', 'train time = ' , (toc-tic)/60)
    print('read out norm = ', np.linalg.norm(wo))
    print('dummy norm = ', np.linalg.norm(wd, axis=0, keepdims=True))

    model_params = {'J':J, 'g':g, 'wf':wf, 'wo':wo, 'wfd':wfd, 'wd':wd, 'wi':wi}
    params['model'] = model_params
    task_prs['counter'] = i
    return x, params


def one_trial_noextend(params, digits_rep, labels, trial, x_ICs, ic_index, seed):

    net_prs = params['network']
    task_prs = params['task']
    model_prs = params['model']
    wo, wd = model_prs['wo'], model_prs['wd']
    wf, wfd, wi = model_prs['wf'], model_prs['wfd'], model_prs['wi']
    J = model_prs['J']
    dt, tau, g = net_prs['dt'], net_prs['tau'], net_prs['g']
    fw_steps = int(1 * task_prs['t_trial'] / net_prs['dt'])

    task = sum_task_experiment(task_prs['n_digits'], 0, 0, 1, task_prs['time_intervals'],
                               net_prs['dt'], task_prs['output_encoding'], [trial], digits_rep, labels,
                               seed)

    exp_mat, target_mat, dummy_mat, i_digits, output_digits = task.experiment()


    x = x_ICs[ic_index, :][np.newaxis].T
    r = np.tanh(x)
    z = np.matmul(wo.T, r)
    zd = np.matmul(wd.T, r)

    x_mat, r_mat = np.zeros([net_prs['N'], fw_steps]), np.zeros([net_prs['N'], fw_steps])
    z_mat = np.zeros([1, fw_steps])

    for i in range(fw_steps):

        x_mat[:, i] = x.reshape(-1)
        r_mat[:, i] = r.reshape(-1)
        z_mat[:, i] = z

        dx = -x + g * np.matmul(J, r) + np.matmul(wf, z) + np.matmul(wi, exp_mat[:, i].reshape(
            [net_prs['d_input'], 1])) + np.matmul(wfd, zd)

        x = x + (dx * dt) / tau

        r = np.tanh(x)
        z = np.matmul(wo.T, r)
        zd = np.matmul(wd.T, r)

    return r_mat, x_mat, z_mat, target_mat


def error_rate(params, x_ICs, digits_rep, labels):
    '''
    evaluate if network has learned the task, by evaluating if RMSE is below train_prs['epsilon']
    Args:
        x_ICs: initial states (N X num_trials)
    '''
    net_prs = params['network']
    task_prs = params['task']
    model_prs = params['model']
    train_prs = params['train']
    num_seeds = 10
    all_trials = task_prs['keep_perms']
    t_resp = int( task_prs['time_intervals']['response'] / net_prs['dt'])

    num_z = x_ICs.shape[0] * len(task_prs['keep_perms']) * num_seeds
    stacked_err = []

    for i in range(num_seeds):
        for ic_index in range(x_ICs.shape[0]):
            for trial in all_trials:
                _, _, zo, z_target = one_trial_noextend(params, digits_rep, labels, trial, x_ICs, ic_index, i)


                rmse = np.sqrt(np.sum((zo[:, -t_resp:] - z_target[:, -t_resp:])**2)/num_z)
                stacked_err.append(rmse)


    err_index0 = np.array(stacked_err) > train_prs['epsilon'][0]
    err_index1 = np.array(stacked_err) > train_prs['epsilon'][1]
    err_index2 = np.array(stacked_err) > train_prs['epsilon'][2]
    err_index3 = np.array(stacked_err) > train_prs['epsilon'][3]
    err_ratio = [sum(err_index0)/num_z, sum(err_index1)/num_z, sum(err_index2)/num_z, sum(err_index3)/num_z ]

    return err_ratio










