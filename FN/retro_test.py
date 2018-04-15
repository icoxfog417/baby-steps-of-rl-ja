import retro

env = retro.make(game='Pong-Atari2600')
env.reset()
while True:
    env.render()
    _obs, _rew, done, _info = env.step(env.action_space.sample())
    if done:
        break
