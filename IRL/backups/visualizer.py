import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm


def describe(episode, name, values, interval=10, round_count=-1):
    mean = np.mean(values[-interval:])
    std = np.std(values[-interval:])
    if round_count > 0:
        mean = np.round(mean, round_count)
        std = np.round(std, round_count)
    desc = "{} is {} (+/-{})".format(name, mean, std)
    print("At episode {}, {}".format(episode, desc))


def plot_values(name, values, interval=10):
    indices = list(range(0, len(values), interval))
    means = []
    stds = []
    for i in indices:
        _values = values[i:(i + interval)]
        means.append(np.mean(_values))
        stds.append(np.std(_values))
    means = np.array(means)
    stds = np.array(stds)
    plt.figure()
    plt.title("{} History".format(name))
    plt.grid()
    plt.fill_between(indices, means - stds, means + stds,
                     alpha=0.1, color="g")
    plt.plot(indices, means, "o-", color="g",
             label="{} per {} episode".format(name.lower(), interval))
    plt.legend(loc="best")
    plt.show()


def plot_grid_rewards(env, Q):
    """
    Show Q-values for FrozenLake-v0.
    To show each action's evaluation,
    a state is shown as 3 x 3 matrix like following.
    XoX Up,
    oco Left, Center(set mean value), Right
    XoX Down
    actions are located on 3 x 3 grid.
    """
    nrow = env.unwrapped.nrow
    ncol = env.unwrapped.ncol
    state_size = 3
    q_nrow = nrow * state_size
    q_ncol = ncol * state_size
    reward_map = np.zeros((q_nrow, q_ncol))

    for r in range(nrow):
        for c in range(ncol):
            s = r * nrow + c
            state_exist = False
            if isinstance(Q, dict) and s in Q:
                state_exist = True
            elif isinstance(Q, (np.ndarray, np.generic)) and s < Q.shape[0]:
                state_exist = True

            if state_exist:
                # In the display map, vertical index reverse.
                _r = 1 + (nrow - 1 - r) * state_size
                _c = 1 + c * state_size
                reward_map[_r][_c - 1] = Q[s][0]  # LEFT = 0
                reward_map[_r - 1][_c] = Q[s][1]  # DOWN = 1
                reward_map[_r][_c + 1] = Q[s][2]  # RIGHT = 2
                reward_map[_r + 1][_c] = Q[s][3]  # UP = 3
                # Center
                reward_map[_r][_c] = np.mean(Q[s])

    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    plt.imshow(reward_map, cmap=cm.RdYlGn, interpolation="bilinear",
               vmax=abs(reward_map).max(), vmin=-abs(reward_map).max())
    ax.set_xlim(-0.5, q_ncol - 0.5)
    ax.set_ylim(-0.5, q_nrow - 0.5)
    ax.set_xticks(np.arange(-0.5, q_ncol, state_size))
    ax.set_yticks(np.arange(-0.5, q_nrow, state_size))
    ax.set_xticklabels(range(ncol + 1))
    ax.set_yticklabels(range(nrow + 1))
    ax.grid(which="both")
    plt.show()
