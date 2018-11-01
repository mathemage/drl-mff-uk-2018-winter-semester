#!/usr/bin/env python3
import numpy as np

import mountain_car_evaluator

if __name__ == "__main__":
	# Fix random seed
	np.random.seed(42)

	# Parse arguments
	import argparse
	parser = argparse.ArgumentParser()
	parser.add_argument("--episodes", default=None, type=int, help="Training episodes.")
	parser.add_argument("--render_each", default=None, type=int, help="Render some episodes.")

	parser.add_argument("--alpha", default=0.5, type=float, help="Learning rate.")
	parser.add_argument("--alpha_final", default=0.01, type=float, help="Final learning rate.")
	parser.add_argument("--epsilon", default=0.5, type=float, help="Exploration factor.")
	parser.add_argument("--epsilon_final", default=0.001, type=float, help="Final exploration factor.")
	parser.add_argument("--gamma", default=1, type=float, help="Discounting factor.")
	args = parser.parse_args()

	# Create the environment
	env = mountain_car_evaluator.environment()

	# TODO: Implement Q-learning RL algorithm.
	#
	# The overall structure of the code follows.
	alpha = args.alpha
	epsilon = args.epsilon
	Q = np.zeros((env.states, env.actions))

	training = True
	while training:
		# Perform a training episode
		state, done = env.reset(), False
		while not done:
			if args.render_each and env.episode and env.episode % args.render_each == 0:
				env.render()

			action = None                                     # TODO choose `action` by epsilon-greedy from Q
			next_state, reward, done, _ = env.step(action)    # take action
			Q[state, action] += alpha * (reward + args.gamma * np.amax(Q[next_state, :], axis=1) - Q[state, action])  # update Q
			state = next_state                                # next state
		Q[state, :] = 0   # action-values at terminal states

	# TODO decay alpha and epsilon

	# Perform last 100 evaluation episodes
