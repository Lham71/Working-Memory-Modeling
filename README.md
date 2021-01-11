# Slow manifolds in recurrent networks encode working memory efficiently and robustly
For mathematical/model details please see a version of the manuscript on arXiv: 
http://github.com - automatic!
[GitHub](http://github.com)

## Train Models
All the results can be reproduced using the codes available in this repository. 
To train thousands of RNN models on cluster, please see submit_jobs.py. This script takes user-defined parameters and submits all jobs automatically. 

Train networks with: 
```
python submit_jobs.py -g 0.8 0.9 1.1 1.2 1.3 1.4 1.5  -pg  0.4 0.6 .07 0.9  -fb 40 30 20 10 5 1 -in 50 -n 3500 -e 1 0.5 1.5 -s 0 1 2 3 -i 'Gauss'


'''
g: chaos level (initial connectivity strength)
pg: sparsity of initial connectivity
fb: feedback strength (eg if fb=40 then the initial feedback weights are divided by 40 in the main script for training (trian_posthoc.py))
in: 1/strength of input weights
n: # of training iterations
e: trial output encoding (ie summation result encoding, here the digits are 0 and 1 so there are 4 trials (00, 01, 10, 11) and there are 3 possible
outcomes for summation that are encoded with 1, 0.5 and 1.5)
s: initialization random seed
i: initial distribution for weight vectors and initial states of RNN (can be either 'Gauss' or 'Uniform')
'''
```
