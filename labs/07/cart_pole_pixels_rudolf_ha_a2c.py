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

import cart_pole_pixels_evaluator

class Network:
	def __init__(self, threads, seed=42):
		# Create an empty graph and a session
		graph = tf.Graph()
		graph.seed = seed
		self.session = tf.Session(graph = graph, config=tf.ConfigProto(inter_op_parallelism_threads=threads,
		                                                               intra_op_parallelism_threads=threads))

	def construct_conv_layers(self, cnn_desc, input, name_prefix):
		depth = len(cnn_desc)
		layers = [None] * (1 + depth)
		layers[0] = input
		for l in range(depth):
			layer_idx = l + 1
			layer_name = "{}_layer{}-{}".format(name_prefix, l, cnn_desc[l])
			specs = cnn_desc[l].split('-')
			# print("...adding layer {} with specs {}".format(layer_name, specs))
			if specs[0] == 'C':
				# - C-filters-kernel_size-stride-padding: Add a convolutional layer with ReLU activation and
				#   specified number of filters, kernel size, stride and padding. Example: C-10-3-1-same
				layers[layer_idx] = tf.layers.conv2d(inputs=layers[layer_idx - 1], filters=int(specs[1]),
				                                     kernel_size=int(specs[2]), strides=int(specs[3]), padding=specs[4],
				                                     activation=tf.nn.relu, name=layer_name)
			if specs[0] == 'M':
				# - M-kernel_size-stride: Add max pooling with specified size and stride. Example: M-3-2
				layers[layer_idx] = tf.layers.max_pooling2d(inputs=layers[layer_idx - 1], pool_size=int(specs[1]),
				                                            strides=int(specs[2]), name=layer_name)
			if specs[0] == 'F':
				# - F: Flatten inputs
				layers[layer_idx] = tf.layers.flatten(inputs=layers[layer_idx - 1], name=layer_name)
			if specs[0] == 'R':
				# - R-hidden_layer_size: Add a dense layer with ReLU activation and specified size. Ex: R-100
				layers[layer_idx] = tf.layers.dense(inputs=layers[layer_idx - 1], units=int(specs[1]), activation=tf.nn.relu,
				                                    name=layer_name)
		return layers

	def construct(self, args, state_shape, num_actions):
		with self.session.graph.as_default():
			self.states = tf.placeholder(tf.float32, [None] + state_shape, name="states")
			self.actions = tf.placeholder(tf.int32, [None], name="actions")
			self.returns = tf.placeholder(tf.float32, [None], name="returns")

			# Add network running inference.
			#
			# For generality, we assume the result is in `self.predictions`.
			#
			# Only this part of the network will be saved, in order not to save
			# optimizer variables (e.g., estimates of the gradient moments).

			# preprocess image
			resized_input = tf.image.resize_images(self.states, size=[40, 40])
			# grayscale_input = tf.image.rgb_to_grayscale(resized_input)
			input = resized_input

			cnn_desc = args.cnn.split(',')

			# policy network - "actor"
			actor_layers = self.construct_conv_layers(cnn_desc, input, name_prefix="actor")
			actor_features = tf.layers.flatten(inputs=actor_layers[-1], name="flattened_features_of_actor")
			logits = tf.layers.dense(actor_features, num_actions)
			self.predictions = tf.nn.softmax(logits)

			# baseline network - "critic"
			critic_layers = self.construct_conv_layers(cnn_desc, input, name_prefix="critic")
			critic_features = tf.layers.flatten(inputs=critic_layers[-1], name="flattened_features_of_critic")
			expanded_baseline = tf.layers.dense(critic_features, 1)
			# - modify the result to have shape `[batch_size]` (you can use for example `[:, 0]`)
			self.baseline = tf.squeeze(expanded_baseline)

			# Saver for the inference network
			self.saver = tf.train.Saver()

			# Training using operation `self.training`.
			# Compute `loss` as a sum of two losses:
			# - sparse softmax cross entropy of `self.actions` and `logits`,
			#   weighted by `self.returns - baseline`. You should not backpropagate
			#   gradient into `baseline` by using `tf.stop_gradient(baseline)`.
			weights = self.returns - tf.stop_gradient(self.baseline)
			loss_actor = tf.losses.sparse_softmax_cross_entropy(
				labels=self.actions,
				logits=logits,
				weights=weights
			)
			# - mean square error of the `self.returns` and `baseline`
			loss_critic = tf.losses.huber_loss(self.returns, self.baseline)
			loss = loss_actor + loss_critic

			global_step = tf.train.create_global_step()
			if args.learning_rate_final:
				decay_rate = (args.learning_rate_final / args.learning_rate)**(1.0 / (args.episodes - 1))
				learning_rate = tf.train.exponential_decay(learning_rate=args.learning_rate, decay_rate=decay_rate,
				                                global_step=global_step, decay_steps=1, staircase=False)
			else:
				learning_rate = args.learning_rate
			self.training = tf.train.AdamOptimizer(learning_rate).minimize(loss, global_step=global_step, name="training")

			# Initialize variables
			self.session.run(tf.global_variables_initializer())

	def predict(self, states):
		return self.session.run(self.predictions, {self.states: states})

	def critic(self, states):
		return self.session.run(self.baseline, {self.states: states})

	def train(self, states, actions, returns):
		self.session.run(self.training, {self.states: states, self.actions: actions, self.returns: returns })

	def save(self, path):
		self.saver.save(self.session, path, write_meta_graph=False, write_state=False)

	def load(self, path):
		self.saver.restore(self.session, path)

if __name__ == "__main__":
	import argparse
	import datetime
	import os
	import re

	# Fix random seed
	np.random.seed(42)

	# Parse arguments
	parser = argparse.ArgumentParser()
	parser.add_argument("--checkpoint", default=None, type=str, help="Checkpoint path.")
	parser.add_argument("--episodes", default=512, type=int, help="Training episodes.")
	parser.add_argument("--gamma", default=1.0, type=float, help="Discounting factor.")
	parser.add_argument("--cnn", default="C-16-5-3-valid,C-24-5-3-valid", type=str,
	                    help="Description of the CNN architecture.")
	parser.add_argument("--learning_rate", default=0.001, type=float, help="Learning rate.")
	parser.add_argument("--learning_rate_final", default=0.001, type=float, help="Final learning rate.")
	parser.add_argument("--render_each", default=128, type=int, help="Render some episodes.")
	parser.add_argument("--threads", default=4, type=int, help="Maximum number of threads to use.")

	parser.add_argument("--evaluate", default=True, type=bool, help="Run evaluation phase.")
	args = parser.parse_args()

	# Create the environment
	env = cart_pole_pixels_evaluator.environment()

	# Construct the network
	network = Network(threads=args.threads)
	network.construct(args, env.state_shape, env.actions)

	# Load the checkpoint if required
	if args.checkpoint:
		# Try extract it from embedded_data
		try:
			import embedded_data
			embedded_data.extract()
		except:
			pass
		network.load(args.checkpoint)
	else:
		best_mean_return = 0
		episode_window = 100
		best_model_path = "cart_pole_pixels/model_best_{}-episode_return".format(episode_window)
		# Training
		for _ in range(args.episodes):
			# Perform episode
			state, done = env.reset(), False
			while not done:
				if args.render_each and env.episode > 0 and env.episode % args.render_each == 0:
					env.render()

				action_probabilities = network.predict([state])[0]
				action = np.random.choice(env.actions, p=action_probabilities)
				next_state, reward, done, _ = env.step(action)

				estimate_of_return = reward
				if not done:
					critic_next_state = network.critic([next_state])
					estimate_of_return += args.gamma * critic_next_state

				# early stop: save the best model so far
				if env.episode >= episode_window:
					mean_return = np.mean(env._episode_returns[- episode_window:])
					if mean_return > best_mean_return:
						print("mean {}-episode return: {} > {} \t -> storing to '{}'".format(
							episode_window,
							mean_return,
							best_mean_return,
							best_model_path)
						)
						best_mean_return = mean_return
						network.save(best_model_path)

				network.train([state], [action], [estimate_of_return])
				state = next_state

		# Save the trained model
		network.save("cart_pole_pixels/model")

		print("'{}' has mean {}-episode return of {}".format(
			best_model_path,
			episode_window,
			best_mean_return
		))

	# Final evaluation: Perform last 100 evaluation episodes
	if args.evaluate:
		print("100 evaluation episodes:")
		for _ in range(100):
			state, done = env.reset(start_evaluate=True), False

			while not done:
				# Compute action `probabilities` using `network.predict` and current `state`
				action_probabilities = network.predict([state])[0]

				# Choose greedy action this time
				action = np.argmax(action_probabilities)
				state, reward, done, _ = env.step(action)
