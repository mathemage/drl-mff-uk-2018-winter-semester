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
import collections

import cart_pole_evaluator
import numpy as np
import tensorflow as tf


class Network:
	def __init__(self, threads, seed=42):
		# Create an empty graph and a session
		graph = tf.Graph()
		graph.seed = seed
		self.session = tf.Session(graph = graph, config=tf.ConfigProto(inter_op_parallelism_threads=threads,
																		 intra_op_parallelism_threads=threads))

	def construct(self, args, state_shape, num_actions, construct_summary=False):
		self.construct_summary = construct_summary
		with self.session.graph.as_default():
			self.states = tf.placeholder(tf.float32, [None] + state_shape)
			self.actions = tf.placeholder(tf.int32, [None])
			self.q_values = tf.placeholder(tf.float32, [None])

			# Compute the q_values
			hidden = self.states
			for _ in range(args.hidden_layers):
				hidden = tf.layers.dense(hidden, args.hidden_layer_size, activation=tf.nn.relu)
			self.predicted_values = tf.layers.dense(hidden, num_actions)

			# Training
			if args.reward_clipping:
				deltas = self.q_values - tf.boolean_mask(self.predicted_values, tf.one_hot(self.actions, num_actions))
				clipped_rewards = tf.clip_by_value(
					deltas,
					-1.0,
					+1.0
				)
				self.loss = tf.losses.mean_squared_error(
          clipped_rewards,
          tf.zeros_like(clipped_rewards)
				)
			else:
				self.loss = tf.losses.mean_squared_error(
					self.q_values,
					tf.boolean_mask(self.predicted_values, tf.one_hot(self.actions, num_actions))
				)
			global_step = tf.train.create_global_step()
			self.training = tf.train.AdamOptimizer(args.learning_rate).minimize(self.loss, global_step=global_step, name="training")

			if construct_summary:
				self.summary_writer = tf.contrib.summary.create_file_writer(args.logdir, flush_millis=10 * 1000)
				self.summaries = {}
				with self.summary_writer.as_default(), tf.contrib.summary.always_record_summaries():
					self.summaries["train"] = [tf.contrib.summary.scalar("train/loss", self.loss)]

			# Initialize variables
			self.session.run(tf.global_variables_initializer())
			if construct_summary:
				with self.summary_writer.as_default():
					tf.contrib.summary.initialize(session=self.session, graph=self.session.graph)

	def copy_variables_from(self, other):
		for variable, other_variable in zip(self.session.graph.get_collection(tf.GraphKeys.GLOBAL_VARIABLES),
											other.session.graph.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)):
			variable.load(other_variable.eval(other.session), self.session)

	def predict(self, states):
		return self.session.run(self.predicted_values, {self.states: states})

	def train(self, states, actions, q_values):
		if self.construct_summary:
			loss, _, _ = self.session.run([self.loss, self.training, self.summaries["train"]], {self.states: states, self.actions: actions, self.q_values: q_values})
			return loss
		else:
			self.session.run(self.training, {self.states: states, self.actions: actions, self.q_values: q_values})

if __name__ == "__main__":
	# Fix random seed
	np.random.seed(42)

	# Parse arguments
	import argparse
	parser = argparse.ArgumentParser()
	parser.add_argument("--batch_size", default=32, type=int, help="Batch size.")
	parser.add_argument("--episodes", default=1000, type=int, help="Episodes for epsilon decay.")
	parser.add_argument("--epsilon", default=0.3, type=float, help="Exploration factor.")
	parser.add_argument("--epsilon_final", default=0.01, type=float, help="Final exploration factor.")
	parser.add_argument("--gamma", default=1.0, type=float, help="Discounting factor.")
	parser.add_argument("--hidden_layers", default=2, type=int, help="Number of hidden layers.")
	parser.add_argument("--hidden_layer_size", default=20, type=int, help="Size of hidden layer.")
	parser.add_argument("--learning_rate", default=0.001, type=float, help="Learning rate.")
	parser.add_argument("--render_each", default=0, type=int, help="Render some episodes.")
	parser.add_argument("--threads", default=1, type=int, help="Maximum number of threads to use.")
	parser.add_argument("--update_every", default=500, type=int, help="Update frequency of target network.")
	parser.add_argument("--replay_buffer_size", default=1024, type=int, help="Maximum size of replay buffer")
	parser.add_argument("--reward_clipping", default=False, type=bool, help="Switch on reward clipping.")
	parser.add_argument("--debug", default=False, type=bool, help="Switch on debug mode.")
	args = parser.parse_args()

	# Create logdir name
	if args.debug:
		import datetime
		import os
		args.logdir = "logs/{}-{}-{}".format(
			os.path.basename(__file__),
			datetime.datetime.now().strftime("%Y-%m-%d_%H%M%S"),
			",".join(map(lambda arg:"{}={}".format(*arg), sorted(vars(args).items())))
		)
		if not os.path.exists("logs"):
			os.mkdir("logs") # TF 1.6 will do this by itself

	# Create the environment
	env = cart_pole_evaluator.environment(discrete=False)

	# Construct the network
	network = Network(threads=args.threads)
	network.construct(args, env.state_shape, env.actions, construct_summary=args.debug)

	# Construct the target network
	target_network = Network(threads=args.threads)
	target_network.construct(args, env.state_shape, env.actions)

	# Replay memory; maxlen parameter can be passed to deque for a size limit,
	# which we however do not need in this simple task.
	replay_buffer = collections.deque(maxlen=args.replay_buffer_size)
	Transition = collections.namedtuple("Transition", ["state", "action", "reward", "done", "next_state"])

	evaluating = False
	training_episodes = 2 * args.episodes
	epsilon = args.epsilon
	update_step = 0
	current_loss = None
	while True:
		# Perform episode
		state, done = env.reset(evaluating), False
		while not done:
			if args.render_each and env.episode > 0 and env.episode % args.render_each == 0:
				env.render()

			# compute action using epsilon-greedy policy.
			if np.random.uniform() > epsilon:
				# You can compute the q_values of a given state:
				q_values = network.predict([state])
				action = np.argmax(q_values)
			else:
				action = np.random.randint(env.actions)

			next_state, reward, done, _ = env.step(action)

			# Append state, action, reward, done and next_state to replay_buffer
			replay_buffer.append(Transition(state, action, reward, done, next_state))

			# If the replay_buffer is large enough,
			replay_size = len(replay_buffer)
			if replay_size >= args.batch_size:
				# perform a training batch of `args.batch_size` uniformly randomly chosen transitions.
				sampled_indices = np.random.choice(replay_size, args.batch_size, replace=False)

				states = []
				actions = []
				rewards = []
				next_states = []
				done_list = []
				for i in sampled_indices:
					transition = replay_buffer[i]
					states.append(transition.state)
					actions.append(transition.action)
					rewards.append(transition.reward)
					next_states.append(transition.next_state)
					done_list.append(transition.done)

				if update_step % args.update_every == 0:
					target_network.copy_variables_from(network)
					if args.debug:
						print("[update step #{}] Copying weights to target net...".format(update_step))
				q_values_in_next_states = target_network.predict(next_states)
				estimates_in_next_states = np.multiply(
					args.gamma * np.max(q_values_in_next_states, axis=-1),
					0,
					where=done_list
				)
				q_values = rewards + estimates_in_next_states

				# After you choose `states`, `actions` and their target `q_values`, train the network
				current_loss = network.train(states, actions, q_values)
				update_step += 1

			state = next_state

		if args.debug:
			print("Loss: {}".format(current_loss))

		# Decide if we want to start evaluating
		evaluating = env.episode > training_episodes

		if not evaluating:
			if args.epsilon_final:
				epsilon = np.exp(np.interp(env.episode + 1, [0, args.episodes], [np.log(args.epsilon), np.log(args.epsilon_final)]))
		else:
			break

	# Perform last 100 evaluation episodes
	for _ in range(100):
		state, done = env.reset(start_evaluate=True), False

		while not done:
			q_values = network.predict([state])
			action = np.argmax(q_values)                 # greedy
			next_state, _, done, _ = env.step(action)
			state = next_state
