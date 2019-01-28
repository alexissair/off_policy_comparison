# Self and efficient off-policy reinforcement learning

## Review by Alexis Sair & Antoine Hoorelbeke

In the paper on this repository, we review the *Safe and efficient off-policy reinforcement
learning* by Rémi Munos, Thomas Stepleton, Anna Harutyunyan and
Marc G. Bellemare. The code here is a first implementation of the algorithm introduced in the paper. 

## Results

The code is designed to be used with gym environements. To simplify things in a first way, we restricted to environements with discrete action and observation spaces. The results present in this README have been produced with the "FrozenLake-v0" environement.

The next graph shows the average reward on 500 episodes for our agent during training. This evaluation is performed during training each 500 episodes. The training is made on 50.000 episodes. Other parameters:  
$ \gamma = 0.95  $  
lambda = 0.9  
n = 100  
alpha=0.01  
epsilon=0.5 with a decay  
gamma=0.8

![frozen-lake]

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
- TB(lambda)
- Importance sampling (IS)
- Q(lambda)
- Retrace (lambda)
- Q-learning

[frozen-lake]: ./results/frozen-lake.png ""
