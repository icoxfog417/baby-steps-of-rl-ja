# Baby Steps of Reinforcement Learning

The step by step guide of Reinforcement Learning with Python.

## Plan before Action: Dynamic Programming

First we think "make plan before action" strategy. It is called "Dynamic Programming".

There are 2 aspect when planning.

1. Estimate the reward on the state.
   * If you know the expected reward on each state, stepping toward higher states led to maximum reward.
2. Brush up the plan.
   * Iteratively update your plan to get more reward, it will be the best plan.

1st is called `Value Iteration`, and 2nd is called `Policy Iteration`.  
This "From the value or policy?" aspect is key point in RL.

Let's experience Dynamic Programming by the cute simulator, and understand its theory from the simple code!

![dp.png](./doc/dp.png)

## Make Plan from Experience

If the number of states is so huge or transition between these is complicated, "planning" cost is high.  
So we have to think the strategy that beginning with action then modifies the plan.

There are 3 point to utilize the experience.

1. The balance between gaining experience and use it to get reward.
   * We don't know the detail of environment.
   * Therefore there are probabilities that more good reward is in states that are not discovered.
   * How to allocate the limited time to experience or its application?
2. The balance between the speed and accuracy for modifying the plan.
   * If we want to fix the plan as soon as possible, we have to estimate the future reward.
   * On the contrary, we can confirm the final reward if we wait until the end of the game, but the fixing is slow.
3. Which one should the experience updates a value or a policy?
   * From the value update aspect, future actions are assumed to select under the optimized policy (in short, optimistic).
   * From the policy update aspect, future actions are selected by its own policy (realistic).

Above problems are linked to representative methods or concept in RL.

1. Epsilon & Greedy method
2. TD learning or Monte Carlo
3. Off policy or On policy

In the examples, following methods are introduced to show the difference of above points.

1. The balance between gaining experience and use it to get reward.
   * [Epsilon & Greedy method](https://github.com/icoxfog417/baby-steps-of-rl/blob/master/EL/notebooks/Epsilon%26Greedy.ipynb)
2. The balance between the speed and accuracy for modifying the plan.
   * As soon as possible (fix after one action) (=TD): [Q-learning](https://github.com/icoxfog417/baby-steps-of-rl/blob/master/EL/notebooks/Q-learning.ipynb)
   * The end of episode: [Monte Carlo](https://github.com/icoxfog417/baby-steps-of-rl/blob/master/EL/notebooks/Monte%20Carlo.ipynb)
3. Which one should the experience updates a value or a policy?
   * From the value update aspect (Off-policy): [Q-learning](https://github.com/icoxfog417/baby-steps-of-rl/blob/master/EL/notebooks/Q-learning.ipynb)
   * From the policy update aspect (On-policy): [SARSA](https://github.com/icoxfog417/baby-steps-of-rl/blob/master/EL/notebooks/SARSA.ipynb)
   * Both of them! : [Actor & Critic](https://github.com/icoxfog417/baby-steps-of-rl/blob/master/EL/notebooks/Actor%26Critic.ipynb)

To evaluate each method, [Frozen Lake](https://gym.openai.com/envs/FrozenLake-v0/) is used (`is_slippery=False` version).  
Each cell represents the state, and North, South, East and West in it shows the estimated reward of each action (`Q[s][a]`).

![Frozen Lake](./doc/frozen_lake.png)

## Represents the Plan as Function

So far we represent the plan by a table. Specifically, the estimated reward of taking the action `a` on a state `s` is Q[s][a]. But as increasing the states and actions, the method takes the high cost.  

So we have to think the way to calculate the value for `s`, `a` without fully recording these. One solution is using a function to represent the relationship between a state and evaluations of each action. And the neural network is one of the most popular "function" to do this recently. 


**contents comming soon...**

Another topics will come

* The week points of (Deep) Reinforcement Learning
* How to overcome week points.
* The adoptation of RL
