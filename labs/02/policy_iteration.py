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

class GridWorld:
	# States in the gridworld are the following:
	# 0 1 2 3
	# 4 x 5 6
	# 7 8 9 10

	# The rewards are +1 in state 3 and -100 in state 6

	# Actions are ↑ → ↓ ←; with probability 80% they are performed as requested,
	# with 10% move 90° CCW is performed, with 10% move 90° CW is performed.
	states = 11

	actions = ["↑", "→", "↓", "←"]

	@staticmethod
	def step(state, action):
		return [GridWorld._step(0.8, state, action),
		        GridWorld._step(0.1, state, (action + 1) % 4),
		        GridWorld._step(0.1, state, (action + 3) % 4)]

	@staticmethod
	def _step(probability, state, action):
		if state >= 5: state += 1
		x, y = state % 4, state // 4
		offset_x = -1 if action == 3 else action == 1
		offset_y = -1 if action == 0 else action == 2
		new_x, new_y = x + offset_x, y + offset_y
		if not(new_x >= 4 or new_x < 0  or new_y >= 3 or new_y < 0 or (new_x == 1 and new_y == 1)):
			state = new_x + 4 * new_y
		if state >= 5: state -= 1
		return [probability, +1 if state == 3 else -100 if state == 6 else 0, state]


def print_value_and_policy():
	for l in range(3):
		for c in range(4):
			state = l * 4 + c
			if state >= 5: state -= 1
			print("        " if l == 1 and c == 1 else "{:-8.2f}".format(value_function[state]), end="")
			print(" " if l == 1 and c == 1 else GridWorld.actions[policy[state]], end="")
		print()


if __name__ == "__main__":
	# Parse arguments
	import argparse
	parser = argparse.ArgumentParser()
	parser.add_argument("--steps", default=10, type=int, help="Number of policy evaluation/improvements to perform.")
	parser.add_argument("--iterations", default=1, type=int, help="Number of iterations in policy evaluation step.")
	parser.add_argument("--gamma", default=1.0, type=float, help="Discount factor.")
	args = parser.parse_args()

	# Start with zero value function and "go North" policy
	value_function = [0] * GridWorld.states
	policy = [0] * GridWorld.states

	# TODO: Implement policy iteration algorithm, with `args.steps` steps of
	# policy evaluation/policy improvement. During policy evaluation, use the
	# current value function and perform `args.iterations` applications of the
	# Bellman equation. Perform the policy evaluation synchronously (i.e., do
	# not overwrite the current value function when computing its improvement).
	for step in range(args.steps):
		# policy evaluation
		for i in range(args.iterations):
			updated_value_function = [0] * GridWorld.states
			for s in range(GridWorld.states):
				for probability, reward, next_state in GridWorld.step(s, policy[s]):
					updated_value_function[s] += probability * (reward + args.gamma * value_function[next_state])
			value_function = updated_value_function

		# policy improvement
		for s in range(GridWorld.states):
			action_values = np.zeros(len(GridWorld.actions))
			for a, _ in enumerate(GridWorld.actions):
				for probability, reward, next_state in GridWorld.step(s, a):
					action_values[a] += probability * (reward + args.gamma * value_function[next_state])
			policy[s] = np.argmax(action_values)

	# Print results
	print_value_and_policy()
