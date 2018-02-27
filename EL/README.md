# Experience makes Plan

There are 3 programs are available.
Use `Taxi-v2` environment except multi-armed-bundit.

* Monte Carlo Control
* Q-learning
* SARASA
* multi-armed-bundit
* Actor-Critic

## How to use experience

### How long experience do you use?

* Till the end of episode: Monte Carlo
* The action just before: Q-learning

### When do you improve the plan from experience?

* In action (Off-policy): Q-learning
* After action (On-policy): SARASA

### The balance between exploration & exploitation

* e-greedy
* multi-armed-bundit (environment from [gym-bandits](https://github.com/JKCooper2/gym-bandits))

## What do you improve by experience?

* The evaluation of state: Q-learning
* The strategy: Actor-Critic

