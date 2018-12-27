# Pythonで学ぶ強化学習 -入門から実践まで-

[Pythonで学ぶ強化学習 -入門から実践まで-](https://www.amazon.co.jp/dp/4065142989/)の実装コードリポジトリです。

## Index

* [Setup](https://github.com/icoxfog417/baby-steps-of-rl-ja#setup)
* [Day1: 強化学習の基礎](https://github.com/icoxfog417/baby-steps-of-rl-ja#day1-%E5%BC%B7%E5%8C%96%E5%AD%A6%E7%BF%92%E3%81%AE%E5%9F%BA%E7%A4%8E)
* [Day2: 強化学習の解法(1): 環境から計画を立てる](https://github.com/icoxfog417/baby-steps-of-rl-ja#day2-%E5%BC%B7%E5%8C%96%E5%AD%A6%E7%BF%92%E3%81%AE%E8%A7%A3%E6%B3%951-%E7%92%B0%E5%A2%83%E3%81%8B%E3%82%89%E8%A8%88%E7%94%BB%E3%82%92%E7%AB%8B%E3%81%A6%E3%82%8B)
* [Day3: 強化学習の解法(2): 経験から計画を立てる](https://github.com/icoxfog417/baby-steps-of-rl-ja#day3-%E5%BC%B7%E5%8C%96%E5%AD%A6%E7%BF%92%E3%81%AE%E8%A7%A3%E6%B3%952-%E7%B5%8C%E9%A8%93%E3%81%8B%E3%82%89%E8%A8%88%E7%94%BB%E3%82%92%E7%AB%8B%E3%81%A6%E3%82%8B)
* [Day4: 強化学習に対するニューラルネットワークの適用](https://github.com/icoxfog417/baby-steps-of-rl-ja#day4-%E5%BC%B7%E5%8C%96%E5%AD%A6%E7%BF%92%E3%81%AB%E5%AF%BE%E3%81%99%E3%82%8B%E3%83%8B%E3%83%A5%E3%83%BC%E3%83%A9%E3%83%AB%E3%83%8D%E3%83%83%E3%83%88%E3%83%AF%E3%83%BC%E3%82%AF%E3%81%AE%E9%81%A9%E7%94%A8)
* [Day5: 深層強化学習の弱点](https://github.com/icoxfog417/baby-steps-of-rl-ja#day5-%E6%B7%B1%E5%B1%A4%E5%BC%B7%E5%8C%96%E5%AD%A6%E7%BF%92%E3%81%AE%E5%BC%B1%E7%82%B9)
* [Day6: 強化学習の弱点を克服するための手法](https://github.com/icoxfog417/baby-steps-of-rl-ja#day6-%E5%BC%B7%E5%8C%96%E5%AD%A6%E7%BF%92%E3%81%AE%E5%BC%B1%E7%82%B9%E3%82%92%E5%85%8B%E6%9C%8D%E3%81%99%E3%82%8B%E3%81%9F%E3%82%81%E3%81%AE%E6%89%8B%E6%B3%95)
* [Day7: 強化学習の活用領域](https://github.com/icoxfog417/baby-steps-of-rl-ja#day7-%E5%BC%B7%E5%8C%96%E5%AD%A6%E7%BF%92%E3%81%AE%E6%B4%BB%E7%94%A8%E9%A0%98%E5%9F%9F)

[Support Content](https://github.com/icoxfog417/baby-steps-of-rl-ja#support-content)

## Setup

サンプルコードをダウンロードするのにGit、実行をするのにPythonの環境が必要です。そのため、以下2つのソフトウェアをダウンロードし、インストールしてください。なお、本書ではPythonの環境を作成するのにMinicondaを使用します。

1. [Git](https://git-scm.com/)
2. [Python (Miniconda)](https://conda.io/miniconda.html)
  * ダウンロードするのは、Python3の方です

インストールが終了したら、まずソースコードのダウンロードを行います。ターミナル/コマンドプロンプトを開き、作業するディレクトリで以下のコマンドを実行してください。

```
git clone https://github.com/icoxfog417/baby-steps-of-rl-ja.git
```

コマンドを実行すると、`baby-steps-of-rl-ja`というディレクトリが作成されていると思います。これで、ダウンロードは完了しました。ダウンロードしたフォルダに移動しましょう。

```
cd baby-steps-of-rl-ja
```

続いて、ソースコードの実行環境を作成します。実行環境を作成するのに、Minicondaをインストールすることで使えるようになる`conda`コマンドを使用します。これから、本書の実行環境である`rl-book`という環境を作成します。

```
conda create -n rl-book python=3.6
activate rl-book  # Mac/Linuxの場合 source activate rl-book
```

`activate`を実行することで、ターミナルの先頭に`(rl-book)`がついたでしょうか。これが、実行環境が有効化されているサインです。本書のソースコードを実行する際は、まず実行環境が有効化されているか=`(rl-book)`が先頭についているか、を確認してください。なお、無効化する際は`deactivate`のコマンドを実行します。

(なお、`python=3.6`と指定しているのは、執筆時点のTensorFlowがPython3.6でしか動かないためです。[#20517](https://github.com/tensorflow/tensorflow/issues/20517)がCloseされればこの指定は不要になります)。

実行環境に、実行に必要なライブラリをインストールします(`(rl-book)`が先頭についているか確認して実行してください)。

```
pip install -r requirements.txt
```

以下のように、`welcome.py`を実行してみてください。ゲーム画面が立ち上がればセットアップは完了です。

```
python welcome.py
```

## Day1: 強化学習の基礎

Day1では、強化学習の位置付けと強化学習の基本的な仕組みについて学びます。強化学習の位置付けとは、強化学習と他の学習方法の関係、また人工知能といったキーワードとの関係などです。強化学習では、「ある環境において得られる報酬が最大になるような行動を学習します」。この一文の中でも「環境」「報酬」「行動」というキーワードが色々出てきましたが、実際に問題を解くには報酬とはなんなのか、行動とはなんなのか・・・など、各概念について明確な定義が必要になります。「基本的な仕組み」とは強化学習における概念とそれらの関係であり、これは「**マルコフ決定過程(Markov Decision Process: MDP)**」という形でうまくまとめられています。

Day1の実装では、強化学習の基本的な仕組みであるMDPを実装することで、その仕組みを理解します。

<p align="center">
  <img src="./doc/mdp.PNG" width=800 alt="mdp.PNG"/>
  <p align="center">MDPの仕組み</p>
</p>

* [MDPの実装](https://github.com/icoxfog417/baby-steps-of-rl-ja/blob/master/DP/environment.py)

## Day2: 強化学習の解法(1): 環境から計画を立てる

Day2では、実際に行動を学習させてみます。Day2における行動の学習は、行動の「価値」を基準に取るべき行動を計画する形になります。そのため、まず評価基準となる「価値」がどのようなものかを学びます。その後に、計画方法である「動的計画法(Dynamic Programming: DP)」を学びます。

<img src="./doc/be.PNG" width=800 alt="be.PNG"/>

* [価値の定義: Bellman Equation](https://github.com/icoxfog417/baby-steps-of-rl-ja/blob/master/DP/bellman_equation.py)
* [価値反復法(Value Iteration)、戦略反復法(Policy Iteration)](https://github.com/icoxfog417/baby-steps-of-rl-ja/blob/master/DP/planner.py)

価値反復法、戦略反復法について実行結果を試せるシミュレーターを用意しています。以下のスクリプトを実行し、立ち上がったサーバーにアクセスしてみてください。

```
python DP/run_server.py
```

http://localhost:8888/

<img src="./doc/application.PNG" width=600 alt="application.PNG"/>

* Areaで行・列を指定し、Drawのボタンを押すことで指定したサイズの迷路を作成することができます。
* 迷路内のセルを選択した後に、Cell SettingにあるTreasure/Danger/Blockのボタンを押すことで、迷路のマスの設定を行うことができます。Treasureはプラスの、Dangerはマイナスの報酬のゴールです。Blockは、移動できないセルになります。
* 迷路の設定ができたら、SimulationにあるValue Iteration/Policy Iterationどちらかのボタンを押すと、ボタンに応じたアルゴリズムで解いた結果が参照できます。

パラメーターを変えながら実行することで、計算の過程を確認してみてください。

## Day3: 強化学習の解法(2): 経験から計画を立てる

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

「経験」とは、具体的には見積もった価値と実際の価値との差異でした。この差異(=誤差)を小さくすることが学習の本質となります。

<img src="./doc/td.PNG" width=800 alt="td.PNG"/>

## Day4: 強化学習に対するニューラルネットワークの適用

Day4では、強化学習をパラメーターを持った関数=ニューラルネットワークで実装する手法を学びます。Day3まで行動は状態/行動のテーブルから決定されていました(Q-Table)。Day4では状態を入力、行動や行動価値を出力とする関数で決定を行います。この関数としてニューラルネットワークを使用します。

* [ニューラルネットワークの仕組み](https://github.com/icoxfog417/baby-steps-of-rl-ja/tree/master/FN/nn_tutorial)
* [価値を関数から算出する](https://github.com/icoxfog417/baby-steps-of-rl-ja/blob/master/FN/value_function_agent.py)
  * [価値をDNNで算出する(DQN)](https://github.com/icoxfog417/baby-steps-of-rl-ja/blob/master/FN/dqn_agent.py)
* [行動を関数から決定する(戦略の関数化)](https://github.com/icoxfog417/baby-steps-of-rl-ja/blob/master/FN/policy_gradient_agent.py)
  * [戦略をDNNで算出する(A2C)](https://github.com/icoxfog417/baby-steps-of-rl-ja/blob/master/FN/a2c_agent.py)

これまでの手法、またDNN(ディープニューラルネットワーク)を利用した手法、双方を含めた手法の系統図は以下のようになっています。

<img src="./doc/rl_ways.PNG" width=800 alt="rl_ways.PNG"/>

## Day5: 深層強化学習の弱点

強化学習に対するDNNの適用は、良いことばかりではありません。具体的には、以下のような問題が現れます。

* サンプル効率が悪い
* 局所最適な行動に陥る、過学習することが多い
* 再現性が低い

Day4の実装では、こうした現実に対応するため、実装を工夫していました。端的には、「１回の学習結果を無駄にしない」ための工夫です。再現性が低いため何度も実験が必要ですが、サンプル効率が悪いため1回の実験に多くの時間がかかります。そのため、「1回」の実験の重みは大きく、これを無駄にしないための工夫が必要です。

具体的には、以下の対策を取っています。

* モジュール化することでテストしやすくする
* ログをしっかりとる

<img src="./doc/train_architecture.PNG" width=800 alt="train_architecture.PNG"/>


## Day6: 強化学習の弱点を克服するための手法

Day6では、Day5で紹介した弱点に対する根本的な対処方法(アルゴリズム的な改良)を扱います。

* サンプル効率が悪い
  * 環境/状態の学習と、そこでの行動方法の学習との分離:「環境認識の改善」
  * [モデルベースとの組み合わせ(Dyna)](https://github.com/icoxfog417/baby-steps-of-rl-ja/tree/master/MM)
* 局所最適な行動に陥る、過学習することが多い
  * 人がある程度誘導してやる: 模倣学習・逆強化学習
  * [模倣学習 (DAgger)](https://github.com/icoxfog417/baby-steps-of-rl-ja/tree/master/IM)
  * [逆強化学習](https://github.com/icoxfog417/baby-steps-of-rl-ja/tree/master/IRL)
* 再現性が低い
  * 新しい学習方法: [進化戦略](https://github.com/icoxfog417/baby-steps-of-rl-ja/tree/master/EV)

なお、サンプル効率については「環境認識の改善」以外に多くの方法があります。以下が、様々な手法をまとめたものになります。

<img src="./doc/sample_improve.PNG" width=600 alt="sample_improve.PNG"/>

## Day7: 強化学習の活用領域

Day7では強化学習を活用する方法を探ります。具体的には、強化学習を活用する観点を整理し、観点ごとの事例、開発を支援ツールなどをまとめています。

<img src="./doc/rl_application.PNG" width=800 alt="rl_application.PNG"/>


## Support Content

プログラミングが初めて、という方のために参考になるコンテンツを用意しています。最近はプログラムを学ぶ書籍などは充実しているため、もちろんそれらで補完して頂いて構いません。

* Python
  * [python_exercises](https://github.com/icoxfog417/python_exercises)
* Git
  * [使い始めるGit](https://qiita.com/icoxfog417/items/617094c6f9018149f41f)
