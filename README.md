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
[誤記、注釈](https://github.com/icoxfog417/baby-steps-of-rl-ja#notation)

## Setup

サンプルコードをダウンロードするのにGit、実行をするのにPythonの環境が必要です。そのため、以下2つのソフトウェアをダウンロードし、インストールしてください。なお、本書ではPythonの環境を作成するのにMinicondaを使用します。

1. [Git](https://git-scm.com/)
2. [Python (Miniconda)](https://conda.io/miniconda.html)
   * ダウンロードするのは、Python3の方です

インストールが終了したら、まずソースコードのダウンロードを行います。ターミナル/コマンドプロンプトを開き、作業するディレクトリで以下のコマンドを実行してください。

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

`activate`を実行することで、ターミナルの先頭に`(rl-book)`がついたでしょうか。これが、実行環境が有効化されているサインです。本書のソースコードを実行する際は、まず実行環境が有効化されているか=`(rl-book)`が先頭についているか、を確認してください。なお、無効化する際は`deactivate`のコマンドを実行します。

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

**Day1's Goals**

* 強化学習と、機械学習、人工知能といったキーワードの関係を理解する
* 強化学習以外の学習法に対する、強化学習のメリット・デメリットを理解する
* 機械学習の基本的な仕組みを理解する

**Summary**

* 強化学習とは?
  * 強化学習 ⊂ 機械学習 ⊂ 人工知能。
  * 機械学習 = 「機械」(=モデル)を「学習」させる手法。
  * 強化学習 = 「学習」方法の一種。
  * 強化学習は、連続した行動を通じて獲得できる「報酬の総和」を最大化することを目的とする。
  * 行動の評価方法と、(評価に基づく)行動の選び方(=戦略)を学習する。
* 強化学習のメリット・デメリット
  * メリット: 評価尺度の定義が難しいタスクでも扱うことができる(行動の評価方法を学習するため)。
  * デメリット: どんな評価を元に、どんな行動を学習するのかはモデル任せになる。
* 強化学習の基本的な仕組み
  * 強化学習では、与えられる「環境」が一定のルールに従っていることを仮定する。そのルールを、 **マルコフ決定過程(Markov Decision Process: MDP)** という。
  * MDPの構成要素とその関係は、以下のように図式化できる。
  * MDPにおける報酬は、「直前の状態と遷移先」に依存する。この報酬を **即時報酬(Immediate reward)** という。
  * 報酬の総和(=即時報酬の合計)は、当然事前には知ることができない。そのため見積りを行うが、見積もった値を **期待報酬(Expected reward)** 、また **価値(Value)** と呼ぶ。
  * 見積もる際に、将来の即時報酬については割り引いて考える。割り引くための係数を **割引率(discount factor)** と呼ぶ。

<p align="center">
  <img src="./doc/mdp.PNG" width=800 alt="mdp.PNG"/>
  <p align="center">MDPの構成要素とその関係</p>
</p>

**Exercises**

* [MDPの実装](https://github.com/icoxfog417/baby-steps-of-rl-ja/blob/master/DP/environment.py)

## Day2: 強化学習の解法(1): 環境から計画を立てる

**Day2's Goals**

* 行動評価の指標となる「価値」の定義を理解する
* 状態の「価値」を動的計画法で学習する手法と実装方法を理解する
* 「戦略」を動的計画法で学習する手法と実装方法を理解する
* モデルベースの手法とモデルフリーの手法の違いを理解する

**Summary**

* 「価値」の定義
  * Day1で定義した「価値」を計算するには、将来の即時報酬が判明している必要がある。
  * 将来の即時報酬は、計算する段階では当然わからない。わからない値に関する計算を持ち越しできるように、式を再帰的に定義する。
  * 得られうる即時報酬の候補はいくつかあり、どれになるかは確率的になる。そのため、報酬の値は期待値(確率x値)で表すようにする(行動確率 x 即時報酬)。
  * 「価値」を再帰的かつ期待値で計算した式を、 **Bellman Equation** と呼ぶ。
* 状態の「価値」の学習と、「戦略」の学習
  * **Bellman Equation** では期待値の計算に戦略(行動確率)を使用する。このため、価値を計算=>得られる価値が高くなるよう、戦略を更新=>戦略が更新されたため、価値を計算し直す=>計算し直した価値で戦略を更新・・・というプロセスを繰り返していくことになる。
  * 動的計画法において、戦略と価値を相互に更新するプロセスを **Policy Iteration** と呼ぶ。
  * 一方、価値が計算できるなら価値が一番高いところを選べばいい、という素朴な考えもある。この場合、価値=戦略となるため、戦略が不要になる。
  * 動的計画法において、価値=戦略とし価値のみ更新するプロセスを **Value Iteration** と呼ぶ。
  * 戦略を持つか(Policyベース)、価値=戦略とするか(Valueベース)は、強化学習の手法分類する際の重要な観点となる
* モデルベースとモデルフリー。
  * 動的計画法では、エージェントを一切動かさずに戦略/価値を学習した。このような芸当が可能なのは、遷移関数と報酬関数が明らかであり、シミュレーションが可能であるため。
  * こうした、環境の情報を元に学習する手法を **モデルベース** の手法と呼ぶ。遷移関数と報酬関数がわかっていることは少ないため、実際は推定を行うことになる。
  * 一方、シミュレーションではなく実際にエージェントを動かすことで得られた経験を元に学習する方法を **モデルフリー** の手法と呼ぶ。モデルの情報(遷移関数/報酬関数)が必要ないため、モデル「フリー」と呼ばれる。
  * 環境が高度になるほどモデルの推定が困難になるため、一般的にはモデルフリーが用いられることが多い。しかし、表現力の高いDNNの登場によりこの限りではなくなっている。また、モデルフリーとモデルベースを併用する試みも行われている。

**Exercises**

* [価値の定義: Bellman Equationの実装](https://github.com/icoxfog417/baby-steps-of-rl-ja/blob/master/DP/bellman_equation.py)
* [価値反復法(Value Iteration)、戦略反復法(Policy Iteration)の実装](https://github.com/icoxfog417/baby-steps-of-rl-ja/blob/master/DP/planner.py)

価値反復法、戦略反復法について実行結果を試せるシミュレーターを用意しています。以下のスクリプトを実行し、立ち上がったサーバーにアクセスしてみてください。

```
python DP/run_server.py
```

http://localhost:8888/

<img src="./doc/application.PNG" width=600 alt="application.PNG"/>

* Areaで行・列を指定し、Drawのボタンを押すことで指定したサイズの迷路を作成できる
* 迷路内のセルを選択した後、Cell SettingにあるTreasure/Danger/Blockのボタンを押すことで、迷路のマスの設定を行うことができる。Treasureはプラスの、Dangerはマイナスの報酬のゴール。Blockは、移動できないセルとなる
* 迷路の設定ができたら、SimulationにあるValue Iteration/Policy Iterationどちらかのボタンを押すと、ボタンに応じたアルゴリズムで解いた結果が参照できる

## Day3: 強化学習の解法(2): 経験から計画を立てる

**Day3's Goals**

* 経験を活用する際の3つの観点を理解する
   1. 経験の蓄積と活用のバランス
   2. 計画の修正を実績から行うか、予測で行うか
   3. 経験を価値、戦略どちらの更新に利用するか
* 各観点における対の関係を理解する
* 各観点を代表する手法の実装方法を身につける

**Summary**

* 「経験」とは
  * 「行動する前」に見積もった価値と、「行動した後」判明した実際の価値との差異となる。
  * 行動すればするほど実際の即時報酬が明らかになり、見積もりに依存する分は少なくなる
  * 「行動する前」の時点と「行動した後」の時点の差、という時刻間の差とも言えるため、これを **TD誤差(Temporal Difference error)** と呼ぶ

<p align="center">
  <img src="./doc/td.PNG" width=600 alt="td.PNG"/>
  <p align="center">経験=TD誤差</p>
</p>

* 経験の蓄積と活用のバランス
  * モデルフリーでは遷移関数/報酬関数が不明なため、「経験」の信頼度を上げるには複数回の試行が必要になる(宝くじを1回買って当選したから、宝くじの当選確率は100%、とはならない)。
  * 行動回数は無限ではないため、限られた行動回数を「経験の信頼度向上」(見積り精度向上)と「経験を信じた行動」にどのように割り当てるのかを決める必要がある。これを **探索と活用のトレードオフ(Exploration-Exploitation Trade-off)** と呼ぶ(探索=信頼度向上、活用=信じた行動)。
  * Epsilonの確率で探索/活用を切り替える手法を、**Epsilon-Greedy法** と呼ぶ。
* 計画の修正を実績から行うか、予測で行うか
  * 「行動した後」は、最短では1回行動した後、最長ではエピソードが終了した後となる。前者を **TD法(TD(0))** 、後者を **Monte Carlo法** と呼ぶ。
  * 「行動した後」を長く取るほど実績に基づいた修正が可能になるが、その分修正のタイミングは遅くなる。実績/タイミングどちらを取るかはトレードオフとなる。
  * TD(0)とMonte Carlo法の間を取ることももちろん可能。「行動した後」を複数ステップ後にする手法を **Multi-step learning** 、ステップ数の異なる経験を組み合わせる手法を **TD(λ)法** と呼ぶ。
* 経験を価値、戦略どちらの更新に利用するか
  * 経験は、価値/戦略(Valueベース/Policyベース)どちらの更新にも利用可能である。
  * TD法に基づき行動の価値の更新を行う手法を **Q-learning** と呼ぶ。"Q"は、行動価値を表す記号としてよく用いられる。状態の価値は"V"、状態における行動の価値は"Q"とされる。
  * 価値を見積る際に、先の行動が戦略により決定されることを前提とする場合を **On-policy** 、Valueベースのように「行動=最大の価値が得られる行動」を前提とする場合を **Off-policy** と呼ぶ(戦略がない=Offのためこう呼ばれる)。
  * Q-learningはOff-policyであり、On-policyとなる手法を**SARSA(State–action–reward–state–action)** と呼ぶ。
  * SARSAでは戦略評価と戦略に同じ"Q"を使用するが、Policy Iterationのように評価と戦略を切り離すこともできる。戦略側をActor、評価側をCriticとして切り離した手法を **Actor-Critic** と呼ぶ。
  * Actor-Criticは、Policyベース(Actor)とValueベース(Critic)の併用とも言える。

修正方法(実績/予測)、修正対象(価値/戦略)、見積り前提(On-policy/Off-policy)の3つの観点で手法をまとめると、以下のようになる。

<table>
  <tr>
    <th rowspan="2"></th>
    <th colspan="2">修正方法</th>
    <th colspan="2">修正対象</th>
    <th colspan="2">見積り前提</th>
  </tr>
  <tr>
    <td>予測</td>
    <td>実績</td>
    <td>価値</td>
    <td>戦略<br></td>
    <td>Off-policy</td>
    <td>On-policy</td>
  </tr>
  <tr>
    <td>Q-learning</td>
    <td>○</td>
    <td></td>
    <td>○</td>
    <td></td>
    <td>○</td>
    <td></td>
  </tr>
  <tr>
    <td>Monte Carlo</td>
    <td></td>
    <td>○</td>
    <td>○</td>
    <td></td>
    <td>○</td>
    <td></td>
  </tr>
  <tr>
    <td>SARSA</td>
    <td>○</td>
    <td></td>
    <td>○</td>
    <td>(○)</td>
    <td></td>
    <td>○</td>
  </tr>
  <tr>
    <td>Actor Critic</td>
    <td>○</td>
    <td></td>
    <td>○</td>
    <td>○</td>
    <td></td>
    <td>○</td>
  </tr>
  <tr>
    <td>Off-policy Actor Critic</td>
    <td>○</td>
    <td></td>
    <td>○</td>
    <td>○</td>
    <td>○</td>
    <td></td>
  </tr>
  <tr>
    <td>On-policy Monte Carlo</td>
    <td></td>
    <td>○</td>
    <td>○</td>
    <td>○</td>
    <td></td>
    <td>○</td>
  </tr>
  <tr>
    <td>Off-policy Monte Carlo</td>
    <td></td>
    <td>○</td>
    <td>○</td>
    <td>○</td>
    <td>○</td>
    <td></td>
  </tr>
</table>

**Exercises**

* 経験の蓄積と活用のバランス
  * [Epsilon-Greedy法](https://github.com/icoxfog417/baby-steps-of-rl-ja/blob/master/EL/notebooks/Epsilon%26Greedy.ipynb)
* 実績から計画を修正するか、予測で行うか
  * [Monte Carlo](https://github.com/icoxfog417/baby-steps-of-rl-ja/blob/master/EL/notebooks/Monte%20Carlo.ipynb)
  * [Temporal Difference](https://github.com/icoxfog417/baby-steps-of-rl-ja/blob/master/EL/notebooks/Q-learning.ipynb)
* 経験を価値、戦略どちらの更新に利用するか
  * [Valueベース & Off policy: Q-learning](https://github.com/icoxfog417/baby-steps-of-rl-ja/blob/master/EL/notebooks/Q-learning.ipynb)
  * [Policyベース & On policy: SARSA](https://github.com/icoxfog417/baby-steps-of-rl-ja/blob/master/EL/notebooks/SARSA.ipynb)
  * [Valueベース & Policyベース: Actor Critic](https://github.com/icoxfog417/baby-steps-of-rl-ja/blob/master/EL/notebooks/Actor%26Critic.ipynb)

## Day4: 強化学習に対するニューラルネットワークの適用

**Day4's Goals**

* 関数として、ニューラルネットワークを適用するメリット
* 価値評価を、パラメーターを持った関数で実装する方法
* 戦略を、パラメーターを持った関数で実装する方法

**Summary**

* 価値評価/戦略の関数化
  * Day3までは、状態における行動の価値をQ[s][a]というテーブルで管理してきた。
  * しかし、このままでは状態数/行動数が多くなった場合に破綻することは目に見えている。テーブルを関数化することが、この組み合わせ爆発に対応するための一つの解法となる。
  * 関数としてニューラルネットワーク/ディープニューラルネットワークを使用する強化学習を特に「深層強化学習」と呼ぶ。
* 関数として、ニューラルネットワークを使用するメリット・デメリット
  * 人間が実際に観測している「状態」に近いデータをエージェントの学習に使用できる。これは、DNNが特徴抽出に優れているためである(画像ならばCNNなど)。
  * ただ、ニューラルネットワークを使うことで学習時間が長くなるなどのデメリットも発生する(詳細はDay5)。
* 価値評価を、パラメーターを持った関数で実装する
  * 状態を受け取り、行動価値を出力する関数(Q-function)を、ニューラルネットワークで実装する。
  * Atariでハイスコアを出したとして話題になった **Deep Q-Network** 以前にもニューラルネットワークを使用した研究はあったが、学習が安定しないという課題があった。Deep Q-Networkは、学習を安定させる3つの工夫( **Experience Reply** 、**Fixed Target Q-Network** 、**報酬のClipping** )を行うことでこの課題を克服している。
  * [Rainbow](https://arxiv.org/abs/1710.02298)は、Deep Q-Networkにさらに効果が得られる6つの工夫を追加した手法となっている。
* 戦略を、パラメーターを持った関数で実装する
  * 戦略の出力は行動確率であり、これは価値のように事前/事後の差分評価ができない。
  * (AとBの選択を行う際、選んだAが思っていたのとどれくらい違うかは評価できるが(価値評価)、Bをとっていたらどうだったのか?というのは時間を巻き戻さないとわからない)。
  * そのため、差分を小さくするのではなく純粋に戦略によって得られる期待価値を最大化する。
  * 期待値は確率X値で計算できた。戦略から得られる期待価値は、「状態への遷移確率」X「状態での行動確率」X「行動によって得られる価値」で計算できる(J(θ))。
  * この期待価値を、勾配法で最大化する手法を **方策勾配法(Policy Gradient)** と呼ぶ。
  * 「行動によって得られる価値」は、Day3で紹介したように予測を使う/実績を使うなど様々なバリエーションがある。
  * 行動の価値から状態の価値をマイナスした、純粋な行動の価値分を **Advantage** と呼ぶ。
  * Advantageの計算に際しては、行動の価値は実績(Monte Carlo)、状態の価値は予測(TD)で計算と分担することもできる。
  * 状態の価値を予測する側をCritic、戦略側をActorとし、Advantageを使った戦略の学習を行う手法を **Advantage Actor Critic (A2C)** と呼ぶ。
  * 一般的に「A2C」と呼ばれる手法はExperience Replyの代わりに分散環境を使用した行動収拾を行う(本書の実装は、Experience Replyベースとなっている)。
  * 方策勾配法は、勾配の更新方法がとてもデリケートである。そのため、あまり大幅な更新が起きないように工夫した手法としてTRPO、PPOがある。
  * A2Cでは行動それぞれの確率を出力しており、このままでは行動数が増えた時に対応できない。
  * これに対し、価値評価のようにベストな行動一つを出力する手法(Deterministic=決定的 なPolicy Gradient= **DPG**)、また行動分布のパラメーターを出力する手法などがある。
* 価値評価か、戦略か
  * 価値評価は、「最大の価値」の行動しか取らないため、価値の値が拮抗している2つの行動があっても少しでも大きい片方を取り続ける(ことを前提とする)。
  * 戦略の場合、拮抗した価値の行動にそれぞれ同じくらいの行動を割り振る、また行動数が多くなっても対応する方法があるなどのメリットがある。
  * ただ、戦略の学習は価値評価の学習に比べ安定しない傾向がある。2018年時点では、Policy Gradientの学習が意図した通りに行われているのかについてもまだよくわかっていない。
  * 既存の手法は、以下のように分類を行うことができる。

<img src="./doc/rl_ways.PNG" width=800 alt="rl_ways.PNG"/>

**Exercises**

* [ニューラルネットワークの仕組み](https://github.com/icoxfog417/baby-steps-of-rl-ja/tree/master/FN/nn_tutorial)
* [価値を関数から算出する](https://github.com/icoxfog417/baby-steps-of-rl-ja/blob/master/FN/value_function_agent.py)
  * [価値をDNNで算出する(DQN)](https://github.com/icoxfog417/baby-steps-of-rl-ja/blob/master/FN/dqn_agent.py)
* [戦略を関数で実装する](https://github.com/icoxfog417/baby-steps-of-rl-ja/blob/master/FN/policy_gradient_agent.py)
  * [戦略をDNNで実装する(A2C)](https://github.com/icoxfog417/baby-steps-of-rl-ja/blob/master/FN/a2c_agent.py)

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

## Notation

注記や誤記について。

* 本書ではValue approximation=状態評価としていますが、これは「価値評価」とするのが適切です。
  * 「Value」は状態価値(V)もあれば、行動価値(Q)もあるためです。Day2までは状態評価ですが、Day3以降は行動価値(Q)の推定であるため推定している内容(行動価値)と言葉(状態評価)が合わなくなっています。
  * 「状態評価」の表記のゆえんは、価値の定義となるBellman Equationの出自が「状態評価」である点です。この表記のまま用語を統一してしまった形になります。この点は、再版の際に修正します。
* Day3で「経験を状態評価、戦略どちらの更新に利用するか: Off policy vs On policy」とありますが、「経験を状態評価、戦略のどちらの更新に使用するか」という点と、「Off policyとOn policy」という観点は別の話になります。
  * 状態評価(前述の通り、これは「価値評価」が適切です)を更新するか、戦略を更新するかというのは「更新対象」の話であり、「On-policyかOff-policyか」というのは価値の見積もりに際しての前提のことです(最良行動前提か、戦略前提か)。
  * そのため、図3-1は「Off-policy/On-policy」ではなく「Valueベース/Policyベース」の方が適切です。
