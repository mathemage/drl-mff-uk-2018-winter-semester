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

import mountain_car_evaluator

if __name__ == "__main__":
    # Fix random seed
    np.random.seed(42)

    # Parse arguments
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", default=10000, type=int, help="Training episodes.")
    parser.add_argument("--render_each", default=None, type=int, help="Render some episodes.")

    parser.add_argument("--alpha", default=0.2, type=float, help="Learning rate.")
    parser.add_argument("--alpha_final", default=0.01, type=float, help="Final learning rate.")
    parser.add_argument("--epsilon", default=0.1, type=float, help="Exploration factor.")
    parser.add_argument("--epsilon_final", default=0.001, type=float, help="Final exploration factor.")
    parser.add_argument("--gamma", default=1, type=float, help="Discounting factor.")
    parser.add_argument("--tiles", default=8, type=int, help="Number of tiles.") # default 8
    args = parser.parse_args()

    # Create the environment
    env = mountain_car_evaluator.environment(tiles=args.tiles)

    # Implement Q-learning RL algorithm, using linear approximation.
    W = np.zeros([env.weights, env.actions])
    epsilon = args.epsilon
    alpha = args.alpha / args.tiles

    evaluating = False
    while not evaluating:
        state, done = env.reset(evaluating), False
        while not done:
            if args.render_each and env.episode and env.episode % args.render_each == 0:
                env.render()

            # Choose `action` according to epsilon-greedy strategy
            if np.random.uniform() > epsilon:
                action = np.argmax(W[state].sum(axis=0))  # epsilon-greedy
            else:
                action = np.random.randint(env.actions)

            next_state, reward, done, _ = env.step(action)

            # TODO: Update W values
            Q_s_a = W[state, action].sum()
            Q_next_s_max = np.amax(W[next_state].sum(axis=0))
            # for i in state:
            #     W[i, action] += alpha * (reward + args.gamma * Q_next_s_max - Q_s_a)
            W[state, action] += alpha * (reward + args.gamma * Q_next_s_max - Q_s_a)

            state = next_state
            if done:
                break

        # Decide if we want to start evaluating

        if not evaluating:
            if args.epsilon_final:
                epsilon = np.exp(np.interp(env.episode + 1, [0, args.episodes], [np.log(args.epsilon), np.log(args.epsilon_final)]))
            if args.alpha_final:
                alpha = np.exp(np.interp(env.episode + 1, [0, args.episodes], [np.log(args.alpha), np.log(args.alpha_final)])) / args.tiles

        if env.episode > args.episodes:
            evaluating = True
            break

    # Perform the final evaluation episodes
    for _ in range(100):
        state, done = env.reset(evaluating), False
        while not done:
            # choose action as a greedy action
            action = np.argmax(W[state].sum(axis=0))  # epsilon-greedy
            state, reward, done, _ = env.step(action)
