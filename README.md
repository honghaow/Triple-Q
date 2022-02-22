## Triple-Q



Author's implementation of the paper: 

**Triple-Q: A Model-Free Algorithm for Constrained Reinforcement Learning with Sublinear Regret and Zero Constraint Violation**

In this paper we proposed the ﬁrst model-free, simulator-free reinforcement learning algorithm for Constrained Markov Decision Processes (CMDPs) with sublinear regret and zero constraint violation.



## A Tabular Case

![](https://github.com/honghaow/Triple-Q/blob/master/env/grid_world.png)

In the tabular case we evaluated our algorithm using a grid-world environment.

Train Triple-Q on this environment by simply running the file ``Triple_Q_tabular.ipynb`` on [Google Colab](https://colab.research.google.com/notebooks/intro.ipynb).



## Deep-Triple-Q



The codes for Deep-Triple-Q are adapted from [Safety Starter Agent](https://github.com/openai/safety-starter-agents) and  [WCSAC](https://github.com/AlgTUDelft/WCSAC).

Triple-Q can also be implemented with neural network approximations and the actor-critic method. 

Train Deep-Triple-Q on the Dynamic Gym benchmark (DynamicEnv) ([Yang et al. (2021)](https://ojs.aaai.org/index.php/AAAI/article/view/17272)) by simply running

```
python ./deep_tripleq/sac/triple_q.py --env 'DynamicEnv-v0' -s 1234 --cost_lim 15 --logger_kwargs_str '{"output_dir":"./temp"}'
```



> **Warning:** If you want to use the Triple-Q algorithm in [Safety Gym](https://github.com/openai/safety-gym), make sure to install Safety Gym according to the instructions on the [Safety Gym repo](https://github.com/openai/safety-gym).



**Deep-Triple-Q  on safe RL with hard  constraints**

Train Deep-Triple-Q on [Pendulum environment](https://gym.openai.com/envs/Pendulumv0/) with hard safety constraints (details can be found in [Cheng et al.](https://arxiv.org/pdf/1903.08792.pdf) ) by running

```
python ./saferl/main_triple_q.py --seed 1234
```

