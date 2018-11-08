# Pythonで学ぶ強化学習 -入門から実践まで-

[Pythonで学ぶ強化学習 -入門から実践まで-]()の実装コードリポジトリです。

## Index

### Day1: 強化学習の基礎

Day1の実装では、強化学習が前提とする問題設定である「マルコフ決定過程(Markov Decision Process: MDP)」の仕組みについて学びます。

* [MDPの実装](https://github.com/icoxfog417/baby-steps-of-rl-ja/blob/master/DP/environment.py)

### Day2: 強化学習の解法(1): 環境から計画を立てる

Day2では、MDPの環境を実際に解いてみます。それにあたり、まず行動の指針となる「価値」を定義する必要がありました。その後に、実際「動的計画法(Dynamic Programming: DP)」を用いて環境を解いてみます。動的計画法はモデルベースと呼ばれる手法の一種で、モデル(=環境の遷移関数と報酬関数)から行動計画を作成します。

* [価値の定義: Bellman Equation](https://github.com/icoxfog417/baby-steps-of-rl-ja/blob/master/DP/bellman_equation.py)
* [価値反復法(Value Iteration)、戦略反復法(Policy Iteration)](https://github.com/icoxfog417/baby-steps-of-rl-ja/blob/master/DP/planner.py)

シミュレーターの実行

```
python DP/run_server.py
```

### Day3: 強化学習の解法(2): 経験から計画を立てる

Day3では経験から計画を立てる方法を学びます。経験から計画を立てる際は、3つの観点がありました。

* 経験の蓄積と活用のバランス
  * [Epsilon-Greedy法](https://github.com/icoxfog417/baby-steps-of-rl-ja/blob/master/EL/notebooks/Epsilon%26Greedy.ipynb)
* 実績から計画を修正するか、予測で行うか
  * [Monte Carlo](https://github.com/icoxfog417/baby-steps-of-rl-ja/blob/master/EL/notebooks/Monte%20Carlo.ipynb)
  * [Temporal Difference](https://github.com/icoxfog417/baby-steps-of-rl-ja/blob/master/EL/notebooks/Q-learning.ipynb)
* 経験を状態評価、戦略どちらの更新に利用するか
  * [On policy: SARSA](https://github.com/icoxfog417/baby-steps-of-rl-ja/blob/master/EL/notebooks/SARSA.ipynb)
  * [Off policy: Q-learning](https://github.com/icoxfog417/baby-steps-of-rl-ja/blob/master/EL/notebooks/Q-learning.ipynb)
  * [Actor Critic](https://github.com/icoxfog417/baby-steps-of-rl-ja/blob/master/EL/notebooks/Actor%26Critic.ipynb)

### Day4: 強化学習に対するニューラルネットワークの適用

Day4では、強化学習をパラメーターを持った関数=ニューラルネットワークで実装する手法を学びます。

### Day5: 深層強化学習の弱点

### Day6: 強化学習の弱点を克服するための手法
