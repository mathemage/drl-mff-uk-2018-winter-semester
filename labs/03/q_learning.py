#!/usr/bin/env python3
import numpy as np

import mountain_car_evaluator

if __name__ == "__main__":
	# Fix random seed
	np.random.seed(42)

	# Parse arguments
	import argparse
	parser = argparse.ArgumentParser()
	parser.add_argument("--episodes", default=400, type=int, help="Training episodes.")
	# parser.add_argument("--episodes", default=10000, type=int, help="Training episodes.")   # TODO uncomment
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
	Q = np.zeros((env.states, env.actions))
	alpha_decay_step = (args.alpha - args.alpha_final) / args.episodes
	epsilon_decay_step = (args.epsilon - args.epsilon_final) / args.episodes

	training = True
	# last_100_rewards = [0] * 100
	while training:
		# linearly decay alpha and epsilon
		alpha = args.alpha - env.episode * alpha_decay_step
		epsilon = args.epsilon - env.episode * epsilon_decay_step
		# print("Ep. {}: alpha {}, epsilon {}".format(env.episode, alpha, epsilon))

		# Perform a training episode
		state, done = env.reset(), False
		while not done:
			if args.render_each and env.episode and env.episode % args.render_each == 0:
				env.render()

			if np.random.uniform() > args.epsilon:
				action = np.argmax(Q[state, :])                 # greedy
			else:
				action = np.random.randint(env.actions)
			next_state, reward, done, _ = env.step(action)    # take action
			Q[state, action] += alpha * (reward + args.gamma * np.amax(Q[next_state, :]) - Q[state, action])  # update Q
			state = next_state                                # next state

		# if (sum(last_100_rewards) / 100 > 492) or env.episode > args.episodes:
		if env.episode > args.episodes:
			break


	# Perform last 100 evaluation episodes
	for _ in range(100):
		state, done = env.reset(start_evaluate=True), False

		while not done:
			action = np.argmax(Q[state, :])                 # greedy
			next_state, _, done, _ = env.step(action)
			state = next_state
