#!/usr/bin/env python3
import sys

import numpy as np

import lunar_lander_evaluator

def m_choose_action(policy, state):
    random_float = np.random.random()
    cnt = 0
    for i in range(len(policy[state][:]) - 1):
        if (random_float >= cnt) and random_float < (cnt + policy[state][i+1]):
            return i
        cnt += policy[state][i]
    return len(policy[state][:]) - 1


if __name__ == "__main__":
    # Fix random seed
    np.random.seed(42)

    # Parse arguments
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", default=None, type=int, help="Training episodes.")
    parser.add_argument("--render_each", default=0, type=int, help="Render some episodes.")

    parser.add_argument("--alpha", default=0.2, type=float, help="Learning rate.")
    parser.add_argument("--alpha_final", default=None, type=float, help="Final learning rate.")
    parser.add_argument("--epsilon", default=0.1, type=float, help="Exploration factor.")
    parser.add_argument("--epsilon_final", default=0.0005, type=float, help="Final exploration factor.")
    parser.add_argument("--gamma", default=0.999, type=float, help="Discounting factor.")
    args = parser.parse_args()

    # Create the environment
    env = lunar_lander_evaluator.environment()

    # The environment has env.states states and env.actions actions.
    N = 8
    epsilon = args.epsilon

    Q = np.zeros((env.states, env.actions,))
    policy = np.ones((env.states, env.actions,))
    # init policy
    for i in range(env.states):
        random_int = np.random.randint(0, env.actions - 1)
        policy[i][:] = (epsilon/env.actions) * policy[i][:]
        policy[i][random_int] = 1 - epsilon + (epsilon/env.actions)

    # The overall structure of the code follows.
    training = True
    while training:

        # To generate expert trajectory, you can use
        #state, trajectory = env.expert_trajectory()


        # Perform a training episode
        state, done = env.reset(), False

        T = sys.maxsize
        stored_states = [state]
        # choose and store action A_0
        action = m_choose_action(policy, state)
        stored_actions = [action]
        stored_rewards = [0]

        while not done:
            if args.render_each and env.episode and env.episode % args.render_each == 0:
                env.render()

            for t in range(0, sys.maxsize):
                if t < T:
                    # take action A_t
                    action = stored_actions[t]
                    next_state, reward, done, _ = env.step(action)
                    # observe and store next reward and state
                    stored_rewards.append(reward)
                    stored_states.append(next_state)

                    # if S_{t+1} id terminal
                    if done == True:
                        T = t + 1
                    else:
                        # choose action A_{t+1}
                        action = m_choose_action(policy, next_state)
                        # store action A_{t+1}
                        stored_actions.append(action)
                tau = t + 1 - N
                if tau >= 0:
                    if t + 1 >= T:
                        G = stored_rewards[T]
                    else:
                        G = stored_rewards[t+1] + args.gamma*np.sum(policy[stored_states[t+1]][:]*Q[stored_states[t+1]][:])

                    for k in range(min(t, T-1), tau + 1 - 1, -1):
                        G = stored_rewards[k] + args.gamma*np.sum([policy[stored_states[k]][a]*Q[stored_states[k]][a] for a in range(env.actions) if a != stored_actions[k]]) + args.gamma*policy[stored_states[k]][stored_actions[k]]*G

                    Q[stored_states[tau]][stored_actions[tau]] = Q[stored_states[tau]][stored_actions[tau]]+args.alpha*(G-Q[stored_states[tau]][stored_actions[tau]])
                    # update policy
                    Q_argmax_action = np.argmax(Q[stored_states[tau]][:])
                    policy[stored_states[tau]][:] = (epsilon/env.actions)*np.ones(env.actions)
                    policy[stored_states[tau]][Q_argmax_action] = 1-epsilon+(epsilon/env.actions)

                if tau == T - 1:
                    break
                # epsilon decay
        if epsilon - 1e-7 > 0:
            epsilon -= 1e-7
        else:
            epsilon = 0
        # if args.epsilon_final:
        #     epsilon
            # epsilon = np.exp(np.interp(env.episode + 1,
            #                            [0, args.episodes],
            #                            [np.log(args.epsilon), np.log(args.epsilon_final)]))
    # Perform last 100 evaluation episodes
    for i in range(101):
        state, done = env.reset(start_evaluate=True), False

        while not done:
            action = Q.argmax(state)

            next_state, reward, done, _ = env.step(action)
            state = next_state