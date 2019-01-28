# Self and efficient off-policy reinforcement learning

## Review by Alexis Sair & Antoine Hoorelbeke

In the paper on this repository, we review the *Safe and efficient off-policy reinforcement
learning* by Rémi Munos, Thomas Stepleton, Anna Harutyunyan and
Marc G. Bellemare. The code here is a first implementation of the algorithm introduced in the paper. 

## Reproduction

It is advised to use a virtual environement to install the python depedencies:

```
$ virtualenv -p python3 env
$ source env/bin/activate
```

Then, install the depedencies:

```
$ pip install -r requirements.txt
```

The next command will show you the different parameters available to test the different algorithms on an environement. 
```
$ python agent/plot.py --help
```
By default, the algorithms that will be used to train an agent are:
- TB(&lambda;)
- Importance sampling (IS)
- Q(&lambda;)
- Retrace(&lambda;)
- Q-learning

## Results

The code is designed to be used with gym environements. To simplify things in a first way, we restricted to environements with discrete action and observation spaces. The results present in this file have been produced with the "FrozenLake-v0" environement.  We used a random behaviour policy.

The next graph shows the comparison of the training performance of agents with different hyperparameters. The score is the average reward on 500 episodes during training, and is calculated each 1000 episodes. The training is made on 30.000 episodes. The parameters used for the agent's training are:  
- &gamma; = 0.95  
- &lambda; = 0.9  
- n = 100  
- &alpha; = 0.1 or &alpha; = 0.01 
- &epsilon;_0 = 0.2 (with decay or not)
- When there is a decay, it is such that for the i-th episode, we have &epsilon; = &epsilon;_0 * i **(-1/3)

![frozen-lake]

We see that the TB(&lambda;) and Retrace(&lambda;) are always amongst the best algorithms. They seem to be more robust to the choice of hyperparameters.


[frozen-lake]: ./results/frozen-lake-crossed.png
