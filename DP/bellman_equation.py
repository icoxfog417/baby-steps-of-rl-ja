def V(s, gamma=0.99):
    V = R(s) + gamma * E_at_next(s)
    return V


def R(s):
    if s == "happy_end":
        return 1
    elif s == "bad_end":
        return -1
    else:
        return 0


def E_at_next(s):
    # If game end, expected value is 0
    if s.endswith("end"):
        return 0

    actions = ["up", "down"]
    values = []
    for a in actions:
        transitions = transition_func(s, a)
        v = 0
        for prob, next_state in transitions:
            v += prob * V(next_state)
        values.append(v)
    return max(values)


def transition_func(s, a):
    """
    Make next state by adding action str to state.
    ex: s = 'state', a = 'up' => 'state_up'
        s = 'state_up', a = 'down' => 'state_up_down'

    If the action count == 5, ends the game.
    """

    if len(s.split("_")) == 5:
        # If up occurs >= 4, happy end!
        up_count = sum([1 if s == "up" else 0 for s in s.split("_")])
        ending = "happy_end" if up_count >= 4 else "bad_end"

        return [(1.0, ending)]
    else:
        opposite = "up" if a == "down" else "down"
        return [
                (0.9, s + "_" + a),
                (0.1, s + "_" + opposite)
            ]


if __name__ == "__main__":
    print(V("state"))
    print(V("state_up_up"))
    print(V("state_down_down"))
