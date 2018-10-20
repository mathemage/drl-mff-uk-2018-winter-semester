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


class MultiArmedBandits():
	def __init__(self, bandits, episode_length):
		self._bandits = []
		for _ in range(bandits):
			self._bandits.append(np.random.normal(0., 1.))
		self._done = True
		self._episode_length = episode_length
		print("Initialized {}-armed bandit, maximum average reward is {}".format(bandits, np.max(self._bandits)))

	def reset(self):
		self._done = False
		self._trials = 0
		return None

	def step(self, action):
		if self._done:
			raise ValueError("Cannot step in MultiArmedBandits when there is no running episode")
		self._trials += 1
		self._done = self._trials == self._episode_length
		reward = np.random.normal(self._bandits[action], 1.)
		return None, reward, self._done, {}


def softmax(v):
	"""Compute softmax values for each sets of scores in v."""
	e_v = np.exp(v - np.max(v))
	return e_v / e_v.sum()


if __name__ == "__main__":
	# Fix random seed
	np.random.seed(42)

	# Parse arguments
	import argparse

	parser = argparse.ArgumentParser()
	parser.add_argument("--bandits", default=10, type=int, help="Number of bandits.")
	parser.add_argument("--episodes", default=1000, type=int, help="Training episodes.")
	parser.add_argument("--episode_length", default=1000, type=int, help="Number of trials per episode.")

	parser.add_argument("--mode", default="greedy", type=str, help="Mode to use -- greedy, ucb and gradient.")
	parser.add_argument("--alpha", default=0, type=float, help="Learning rate to use (if applicable).")
	parser.add_argument("--c", default=1., type=float, help="Confidence level in ucb.")
	parser.add_argument("--epsilon", default=0.1, type=float, help="Exploration factor (if applicable).")
	parser.add_argument("--initial", default=0, type=float, help="Initial value function levels.")
	args = parser.parse_args()

	env = MultiArmedBandits(args.bandits, args.episode_length)

	average_rewards = []
	for episode in range(args.episodes):
		env.reset()

		# Done: Initialize required values (depending on mode).
		if args.mode == "gradient":
			h = np.zeros(args.bandits)
		else:  # greedy or UCB
			q = args.initial * np.ones(args.bandits)
			n = np.zeros(args.bandits)

		average_rewards.append(0)
		done = False
		while not done:
			# Done: Action selection according to mode
			if args.mode == "greedy":
				if np.random.uniform() > args.epsilon:
					action = np.argmax(q)
				else:
					action = np.random.randint(args.bandits)
			elif args.mode == "ucb":
				if 0 in n:
					action = np.argmin(n)
				else:
					ucb_sum = args.c * np.sqrt(np.log(episode) / n)
					action = np.argmax(q + ucb_sum)
			elif args.mode == "gradient":
				pi = softmax(h)
				action = np.random.choice(args.bandits, p=pi)

			_, reward, done, _ = env.step(action)
			average_rewards[-1] += reward / args.episode_length

			# Done: Update parameters
			if args.mode == "gradient":
				one_hot = np.zeros(args.bandits)
				one_hot[action] = 1
				h += args.alpha * reward * (one_hot - pi)
			else:  # greedy or UCB
				n[action] += 1
				step_size = 1 / n[action] if args.alpha == 0 else args.alpha
				q[action] += step_size * (reward - q[action])

	# Print out final score as mean and variance of all obtained rewards.
	print("Final score: {}, variance: {}".format(np.mean(average_rewards), np.var(average_rewards)))
	with open("results.txt", "a") as output_file:
		print("{}".format(np.mean(average_rewards)), file=output_file)
