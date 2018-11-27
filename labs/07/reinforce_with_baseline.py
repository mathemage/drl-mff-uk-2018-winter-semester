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
import tensorflow as tf

import cart_pole_evaluator

class Network:
	def __init__(self, threads, seed=42):
		# Create an empty graph and a session
		graph = tf.Graph()
		graph.seed = seed
		self.session = tf.Session(graph = graph, config=tf.ConfigProto(inter_op_parallelism_threads=threads,
		                                                               intra_op_parallelism_threads=threads))

	def construct(self, args, state_shape, num_actions):
		with self.session.graph.as_default():
			self.states = tf.placeholder(tf.float32, [None] + state_shape)
			self.actions = tf.placeholder(tf.int32, [None])
			self.returns = tf.placeholder(tf.float32, [None])

			# Start with self.states and
			# - add a fully connected layer of size args.hidden_layer and ReLU activation
			hidden_actor = tf.layers.dense(self.states, args.hidden_layer, activation=tf.nn.relu)
			# - add a fully connected layer with num_actions and no activation, computing `logits`
			logits = tf.layers.dense(hidden_actor, num_actions)
			# - compute `self.probabilities` as tf.nn.softmax of `logits`
			self.probabilities = tf.nn.softmax(logits)

			# Compute `baseline`, by starting with a fully connected layer processing `self.states` and
			# - add a fully connected layer of size args.hidden_layer and ReLU activation
			hidden_critic = tf.layers.dense(self.states, args.hidden_layer, activation=tf.nn.relu)
			# - add a fully connected layer with 1 output and no activation
			expanded_baseline = tf.layers.dense(hidden_critic, 1)
			# - modify the result to have shape `[batch_size]` (you can use for example `[:, 0]`)
			baseline = tf.squeeze(expanded_baseline)

			# Compute `loss` as a sum of two losses:
			# - sparse softmax cross entropy of `self.actions` and `logits`,
			#   weighted by `self.returns - baseline`. You should not backpropagate
			#   gradient into `baseline` by using `tf.stop_gradient(baseline)`.
			loss_actor = tf.losses.sparse_softmax_cross_entropy(
				labels=self.actions,
				logits=logits,
				weights=self.returns - tf.stop_gradient(baseline)
			)
			# - mean square error of the `self.returns` and `baseline`
			loss_critic = tf.losses.mean_squared_error(self.returns, baseline)
			loss = loss_actor + loss_critic

			global_step = tf.train.create_global_step()
			self.training = tf.train.AdamOptimizer(args.learning_rate).minimize(loss, global_step=global_step, name="training")

			# Initialize variables
			self.session.run(tf.global_variables_initializer())

	def predict(self, states):
		return self.session.run(self.probabilities, {self.states: states})

	def train(self, states, actions, returns):
		self.session.run(self.training, {self.states: states, self.actions: actions, self.returns: returns})

if __name__ == "__main__":
	# Fix random seed
	np.random.seed(42)

	# Parse arguments
	import argparse
	parser = argparse.ArgumentParser()
	parser.add_argument("--batch_size", default=1, type=int, help="Number of episodes to train on.")
	parser.add_argument("--episodes", default=200, type=int, help="Training episodes.")
	parser.add_argument("--gamma", default=1.0, type=float, help="Discounting factor.")
	parser.add_argument("--hidden_layer", default=20, type=int, help="Size of hidden layer.")
	parser.add_argument("--learning_rate", default=0.05, type=float, help="Learning rate.")
	parser.add_argument("--render_each", default=0, type=int, help="Render some episodes.")
	parser.add_argument("--threads", default=1, type=int, help="Maximum number of threads to use.")
	args = parser.parse_args()

	# Create the environment
	env = cart_pole_evaluator.environment(discrete=False)

	# Construct the network
	network = Network(threads=args.threads)
	network.construct(args, env.state_shape, env.actions)

	# Training
	for _ in range(args.episodes // args.batch_size):
		batch_states, batch_actions, batch_returns = [], [], []
		for _ in range(args.batch_size):
			# Perform episode
			states, actions, rewards = [], [], []
			state, done = env.reset(), False
			while not done:
				if args.render_each and env.episode > 0 and env.episode % args.render_each == 0:
					env.render()

				# Compute action probabilities using `network.predict` and current `state`
				action_probabilities = network.predict([state])[0]

				# Choose `action` according to `probabilities` distribution (np.random.choice can be used)
				action = np.random.choice(env.actions, p=action_probabilities)

				next_state, reward, done, _ = env.step(action)

				states.append(state)
				actions.append(action)
				rewards.append(reward)

				state = next_state

			# Compute returns by summing rewards (with discounting)
			returns = []
			G = 0
			for reward in reversed(rewards):
				G *= args.gamma
				G += reward
				returns.append(G)
			returns.reverse()

			# Add states, actions and returns to the training batch
			batch_states.extend(states)
			batch_actions.extend(actions)
			batch_returns.extend(returns)

		# Train using the generated batch
		network.train(batch_states, batch_actions, batch_returns)

	# Final evaluation
	while True:
		state, done = env.reset(True), False
		while not done:
			# Compute action `probabilities` using `network.predict` and current `state`
			action_probabilities = network.predict([state])[0]

			# Choose greedy action this time
			action = np.argmax(action_probabilities)
			state, reward, done, _ = env.step(action)
