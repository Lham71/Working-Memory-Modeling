# Slow manifolds in recurrent networks encode working memory efficiently and robustly
For mathematical/model details please see a version of the manuscript on arXiv: 

## Train Models
All the results can be reproduced using the codes available in this repository. 
To train thousands of RNN models on cluster, please see submit_jobs.py. This script takes user-defined parameters and submits all jobs automatically. 

Train networks with: 
```
python submit_jobs.py -g 0.8 0.9 1.1 1.2 1.3 1.4 1.5  -pg  0.4 0.6 .07 0.9  -fb 40 30 20 10 5 1 -in 50 -n 3500 -e 1 0.5 1.5 -s 0 1 2 3 -i 'Gauss'

```

g: chaos level (initial connectivity strength)
pg: sparsity of initial connectivity
n_train: # of training iterations
encoding: trial output encoding (ie summation result encoding)
seed: random seed
init_dist: initial distribution for weight vectors and initial states of RNN
