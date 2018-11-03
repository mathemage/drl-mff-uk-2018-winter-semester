#!/usr/bin/env python3
#
# All team solutions **must** list **all** members of the team.
# The members must be listed using their ReCodEx ids anywhere
# in the first comment block in the source file, i.e., in the first
# consecutive range of lines beginning with `#`.
#
# You can find out ReCodEx id on URL when watching ReCodEx profile.
# The id has the following format: 01234567-89ab-cdef-0123-456789abcdef.
#
# 090fa5b6-d3cf-11e8-a4be-00505601122b (Jan Rudolf)
# 08a323e8-21f3-11e8-9de3-00505601122b (Karel Ha)
#

import numpy as np
import gym

if __name__ == "__main__":
    # Fix random seed
    np.random.seed(42)

    # Parse arguments
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", default=1000, type=int, help="Training episodes.")
    args = parser.parse_args()

    # Create the environment
    env = gym.make("FrozenLake-v0")
    env.seed(42)
    states = env.observation_space.n
    actions = env.action_space.n

    # Behaviour policy is uniformly random.
    # Target policy uniformly chooses either action 1 or 2.
    V = np.zeros(states)
    C = np.zeros(states)

    for _ in range(args.episodes):
        state, done = env.reset(), False

        rhos = []
        rewards = []
        flag_compute_importance_sampling = True

        # Generate episode
        episode = []
        while not done:
            action = np.random.choice(actions)
            next_state, reward, done, _ = env.step(action)
            if action == 1 or action == 2:
                rhos.append(0.5/0.25)
                episode.append((state, action, reward))
                rewards.append(reward)
            else:
                # flag_compute_importance_sampling = False
                rhos.append(0)
                episode.append((state, action, reward))
                rewards.append(reward)
                # break

            state = next_state

        # TODO: Update V using weighted importance sampling.
        idx = 0
        passed_states = list()
        for (state, action, reward) in episode:
            if state not in passed_states: # zkontrolovat, ze je mozne
                return_episode = np.sum(rewards[idx:])
                W_episode = np.prod(rhos[idx:])
                if (C[state] + W_episode) != 0:
                    C[state] = C[state] + W_episode
                    V[state] = V[state] + (W_episode/C[state]) * (return_episode - V[state])
                # passed_states.append(state)
            idx += 1

    # Print the final value function V
    for row in V.reshape(4, 4):
        print(" ".join(["{:5.2f}".format(x) for x in row]))
