'''
class to construct the experiment -- sequential pattern matching task
written in Python 3.8.3
@ Elham
'''

import numpy as np
import random
from collections import deque
from itertools import combinations, permutations

class sum_task_experiment:

    def __init__(self, digit_size, train_trials_size, ex_train_size, test_trials_size, t_intervals, dt, output_enc, keep_perms, dig_rep, labels,
                 seed):
        ''' Creates SPM task object
        Args:
        digit_size: number of digits in MNIST dataset
        train_trials_size: number trials for training
        ex_train_size: number of extra training trials ( set to zero)
        test_trials_size: number of trials for test phase
        t_intervals: dictionary that contains task time intervals
        dt: Euler integration time step
        output_enc: trial outputs that network should learn (0.5, 1, 1.5)
        keep_perms: tuple of trials  [(0,1), (1,0), (1,1), (0,0)]
        dig_rep: MNIST digit representations
        labels: MNIST labels
        seed: random seed
        '''
        self.digit_size = digit_size 
        self.train_size = train_trials_size
        self.extra_train_size = ex_train_size
        self.test_size = test_trials_size
        self.trial_size = train_trials_size + ex_train_size + test_trials_size
        self.t_intervals = t_intervals
        self.t_trial = sum(self.t_intervals.values()) + self.t_intervals['stim_on'] + self.t_intervals['stim_off']
        self.dt = dt
        self.output_encoding = output_enc
        self.keep_perms = keep_perms
        self.dig_rep = dig_rep
        self.labels = labels
        self.seed = seed
        self.rng = np.random.RandomState(self.seed)
        self.fixate_dim = 1
        self.fixate_value = 0.5
        self.eps = np.hstack((self.rng.rand(self.train_size),
                              self.rng.rand(self.extra_train_size), self.rng.rand(self.test_size))) / 2  # to get different samples over each trial
        self.stim_dim = dig_rep.shape[1]
        self.output_dim = 1


    def get_each_dig_mean(self):
        """
        Returns the mean for each digit
        """
        digits = list(set(self.labels))

        mean = np.zeros([len(digits), 2])

        for i, dig in enumerate(digits):
            index = np.where(self.labels == dig)
            mean_dig = np.mean(self.dig_rep[index[0]], axis=0)
            dig_sample = self.dig_rep[index[0]]
            dig_label = self.labels[index[0]]

            mean[i, :] = mean_dig

        return mean, dig_sample, dig_label


    def all_perms(self):
        '''
        Return: all possible combination of 2 digits from MNIST
        '''
        all_dig = list(range(self.digit_size + 1))
        perms = list(permutations(all_dig, 2))
        all_perms = perms + [(0, 0), (1, 1), (2, 2), (3, 3), (4, 4), (5, 5), (6, 6), (7, 7), (8, 8), (9, 9)]
        for tuples in self.keep_perms:
            all_perms.remove(tuples)

        return all_perms


    def generate_equal_permutated(self, perms, data_size):
        ''' For given trials, generate sequence of trials
        perms: trials
        data_size: number of trials
        Return: array that contains tuples of trials
        '''
        if data_size % len(perms) != 0:
            raise ValueError(
                'number of trials {} is not divisible by number of possible summation inputs {}'.format(data_size,
                                                                                                        len(perms)))
        n = data_size // len(perms)
        seq = []
        for i in range(len(perms)):
            seq += [perms[i]] * n


        #self.rng.shuffle(seq)
        np.random.seed(self.seed)
        indecies = np.random.permutation(len(seq))
        seq = np.asarray(seq)
        seq = seq[indecies].tolist()

        return seq


    def get_encode_potential_ans(self, target_num=0):
        '''
        Returns potential answer for '+' and '-' and encode target into output hot-vector
        '''
        num_list = list(range(self.digit_size + 1))
        self_add = [el + el for el in num_list]
        comb = list(combinations(num_list, 2))
        addition = [el[0] + el[1] for el in comb] + self_add
        # subtraction = [el[0]-el[1] for el in comb] + [el[1]-el[0] for el in comb]+ [0]
        subtraction = []

        potential_target = list(set(addition + subtraction))
        encoded_target = np.zeros([len(potential_target), 1])

        if target_num not in potential_target:
            raise ValueError(
                'target_num {} does not exist in potential answers {}'.format(target_num, potential_target))

        index = potential_target.index(target_num)
        encoded_target[index] = 1

        return potential_target, encoded_target


    def expand(self, on_intv, off_intv, input_dim, input_type=None, second_dig=None, digits_mat=None,
               queue=None, labels=None, eps=None):
        '''
        takes the on and off interval of each input type and expand it by dt, and assign value to each input_type
        input_type : 'fixate', 'cue+', 'cue-', 'stimulus1', 'stimulus2', 'response'

        Returns addition/subtraction value (int) and input_mat (input_dim x t_intv)
        '''

        t_intv = (on_intv + off_intv) / self.dt
        input_mat = np.zeros([input_dim, int(t_intv)])

        if input_type == 'fixate':
            input_mat[:, 0:int(on_intv / self.dt)] = self.fixate_value
            num = None
            queue = None
            dig2 = None

        elif input_type == 'cue+':
            input_mat[:, 0:int(on_intv / self.dt)] = 1.
            num = None
            queue = None
            dig2 = None

        elif input_type == 'cue-':
            input_mat[:, 0:int(on_intv / self.dt)] = -1.
            num = None
            queue = None
            dig2 = None

        elif input_type == 'stimulus1':
            dig = queue.popleft()
            dig1, dig2 = dig[0], dig[1]
            indx_ = np.where(labels == dig1)
            indx = indx_[0]
            random.shuffle(indx, lambda: eps)
            np.random.seed(self.seed)
            input_mat[:, 0:int(on_intv / self.dt)] = digits_mat[np.random.choice(indx, int(on_intv / self.dt))].T
            num = dig1


        elif input_type == 'stimulus2':
            dig2 = second_dig
            indx_ = np.where(labels == dig2)
            indx = indx_[0]
            random.shuffle(indx, lambda: eps - 0.05)
            np.random.seed(self.seed)
            input_mat[:, 0:int(on_intv / self.dt)] = digits_mat[np.random.choice(indx, int(on_intv / self.dt))].T
            num = dig2

        elif input_type == 'delay':
            input_mat[:, 0:int(on_intv / self.dt)] = 0.
            num = None
            queue = None
            dig2 = None

        elif input_type == 'response':
            input_mat = input_mat
            num = None
            queue = None
            dig2 = None

        return num, input_mat, queue, dig2


    def generate_trial(self, t_intv, digit_mat, labels, queue=None, cue_in=None, eps=None):
        ''' generates a single trial
        Return: depending on cue returns target_digit (int) and trial (t_trial x input_dim)
        '''
        input_dim = digit_mat.shape[1]  # 2

        _, fixate, _, _ = self.expand(t_intv['fixate_on'], t_intv['fixate_off'], input_dim, input_type='fixate')

        if cue_in == '+':
            _, cue, _, _ = self.expand(t_intv['cue_on'], t_intv['cue_off'], input_dim, input_type='cue+')
        elif cue_in == '-':
            _, cue, _, _ = self.expand(t_intv['cue_on'], t_intv['cue_off'], input_dim, input_type='cue-')

        num1, stim1, que, secDig = self.expand(t_intv['stim_on'], t_intv['stim_off'], input_dim, input_type='stimulus1',
                                               digits_mat=digit_mat, queue=queue, labels=labels, eps=eps)

        num2, stim2, que, _ = self.expand(t_intv['stim_on'], t_intv['stim_off'], input_dim, input_type='stimulus2',
                                          second_dig=secDig, digits_mat=digit_mat, queue=queue, labels=labels, eps=eps)

        _, delay, _, _ = self.expand(t_intv['delay_task'], 0, input_dim, input_type='delay')

        _, response, _, _ = self.expand(0, t_intv['response'], input_dim, input_type='response')

        fix_cue = np.concatenate((fixate, cue), axis=1)
        stim1_stim2 = np.concatenate((stim1, stim2), axis=1)
        stim1_stim2_delay = np.concatenate((stim1_stim2, delay), axis=1)
        fix_cue_stim1_stim2_delay = np.concatenate((fix_cue, stim1_stim2_delay), axis=1)
        trial = np.concatenate((fix_cue_stim1_stim2_delay, response), axis=1)

        if cue_in == '+':
            target_dig = num1 + num2
        elif cue_in == '-':
            target_dig = num1 - num2

        return num1, num2, target_dig, trial


    def experiment(self):
        """
        generates a sequence of trials equal to num_trials
        and generates encoded target signals
        Returns: exp_mat (num_input x t_exp) and target_mat (num_output x t_exp)
        """

        t_intv = self.t_intervals
        rep_mean, _, _ = self.get_each_dig_mean()
        #print('mean', rep_mean[:2,:])
        removed_perms = self.all_perms()
        train_seq = self.generate_equal_permutated(perms=self.keep_perms, data_size=self.train_size)
        extra_train_seq = self.generate_equal_permutated(perms=self.keep_perms, data_size=self.extra_train_size)
        test_seq = self.generate_equal_permutated(perms=self.keep_perms, data_size=self.test_size)
        num_seq = train_seq + extra_train_seq + test_seq
        que_seq = deque(num_seq)

        input_dim = self.dig_rep.shape[1]
        exp_length = (self.trial_size * self.t_trial) / self.dt
        trial_len = int(self.t_trial / self.dt)
        exp_mat = np.zeros([input_dim, int(exp_length)])
        target_mat = np.zeros([self.output_dim, int(exp_length)])
        fixate_mat = np.ones([self.fixate_dim, int(exp_length)]) * self.fixate_value
        dummy_mat = np.zeros([input_dim, int(exp_length)])

        input_digits = []
        output_digits = []

        i = 0
        while i <= self.trial_size - 1:
            cue_in = '+'
            eps_ = self.eps[i]
            num1, num2, target_digit, trial = self.generate_trial(t_intv=t_intv, digit_mat=self.dig_rep,
                                                                  labels=self.labels,
                                                                  queue=que_seq, cue_in=cue_in, eps=eps_)
            if (num1, num2) in removed_perms:
                i -= 1

            else:
                input_digits.append((cue_in, (num1, num2)))
                output_digits.append(target_digit)

                exp_mat[:, i * trial_len: i * trial_len + trial_len] = trial

                if target_digit == 0:

                    target_digit = self.output_encoding[0]

                elif target_digit == 1:

                    target_digit = self.output_encoding[1]

                elif target_digit == 2:

                    target_digit = self.output_encoding[2]

                elif target_digit == 3:

                    target_digit = self.output_encoding[3]

                elif target_digit == 4:

                    target_digit = self.output_encoding[4]

                target_mat[:,
                (i * trial_len) + trial_len - int(t_intv['response'] / self.dt):(i * trial_len) + trial_len] = \
                    target_digit


                fixate_mat[:,
                (i * trial_len) + trial_len - int(t_intv['response'] / self.dt):(i * trial_len) + trial_len] = 0.

                fix_stim1on_dur = (t_intv['fixate_on'] + t_intv['fixate_off'] + t_intv['stim_on']) / self.dt
                delay1_dur = (t_intv['stim_off']) / self.dt
                fix_stim2on_dur = fix_stim1on_dur + delay1_dur + (t_intv['stim_on'] / self.dt)
                resp_delay_dur = (t_intv['delay_task'] + t_intv['response']) / self.dt

                dummy_mat[:,
                (i * trial_len) + int(fix_stim1on_dur): (i * trial_len) + int(fix_stim1on_dur + delay1_dur)] = \
                    np.reshape(rep_mean[num1, :], [input_dim, 1])
                dummy_mat[:,
                (i * trial_len) + int(fix_stim2on_dur): (i * trial_len) + trial_len - int(resp_delay_dur)] = \
                    np.reshape(rep_mean[num2, :], [input_dim, 1])

            i += 1

        return exp_mat, target_mat, dummy_mat, input_digits, output_digits
