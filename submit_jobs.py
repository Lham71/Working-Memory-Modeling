'''
Python script that takes arguments (hyperparameters of RNN) and automatically writes job files for submission on WashU cluster

written in Python 3.8.3
@ Elham
'''

import os
import sys
import argparse
import subprocess
import itertools
import json
#'g':1.2, 'pg':0.1, 'n_train':1000, 'encoding':[0.5,1.5,1], 'seed':0, 'init_dist':'Gauss'

parser = argparse.ArgumentParser()
parser.add_argument('-g', nargs='+', type=float)
parser.add_argument('-pg', nargs='+', type=float)
parser.add_argument('-fb', '--fb_var', nargs='+', type=float)
parser.add_argument('-in','--input_var', nargs='+', type=float)
parser.add_argument('-n','--n_train', nargs='+', type=int)
parser.add_argument('-e', '--encoding', nargs='+', type=float)
parser.add_argument('-s', '--seed', nargs='+', type=int)
parser.add_argument('-i', '--init_dist', nargs='+', type=str)

arg = parser.parse_args()

sbatchpath = './'
scratchpath = '/scratch/elham/results3500c/' #outputfile
#print('path', scratchpath)

def write_jobfiles(cmd, jobname, sbatchpath, scratchpath, nodes=1, ppn=1, gpus=0, mem=32):
    jobfile = os.path.join(sbatchpath, jobname + '.pbs')
    logname = os.path.join('log', jobname)
    with open(jobfile, 'w') as f:
        f.write(
            '#! /bin/bash\n'
            + '\n'
            + '#PBS -N {}\n'.format(jobname)
            #+ '#PBS -M elham@wustl.edu\n'
            #+ '#PBS -m abe\n'
            + '#PBS -l nodes={}:ppn={},mem={}gb,walltime=23:30:00\n'.format(nodes, ppn, mem)
            + '#PBS -o {}{}.o\n'.format(scratchpath, jobname)
	    + '#PBS -e {}{}.e\n'.format(scratchpath, jobname)
	    + 'cd ./rnn\n'
	    + 'export PATH=/export/Anaconda3-2020.02/bin:$PATH\n'
	    + 'source activate myenv\n'
            + '{}\n'.format(cmd)
            + '{} >> {}.o 1>&1\n'.format(cmd, 'all_logs')
	    + 'pwd\n'
	    + 'echo $PATH\n'
            + 'echo {} >> {}.log 2>&1\n'.format(jobname,'all_names' )
	    + 'exit 0;\n'
        )

    return jobfile


def get_params(**kwargs):
   

    all = list()
    kwargs['encoding']=[kwargs['encoding']]
    keys = list(kwargs)
    
    for values in itertools.product(*map(kwargs.get, keys)):
       
        all_param = dict(zip(keys, values))

        all.append(all_param)

    return all


all = get_params(**vars(arg))

for param in all:
   
    jobname =  '_'.join(['{}'.format(val) if type(val) != list else '{}'.format(''.join([str(s) for s in val])) for key, val in param.items()])
    jparam = json.dumps(param)
    cmd = 'python train_posthoc_clst.py -d ' + '\''+ str(jparam) + '\''
    jobfile = write_jobfiles(cmd, jobname, sbatchpath, scratchpath, gpus=0)
    subprocess.call(['qsub', jobfile])













