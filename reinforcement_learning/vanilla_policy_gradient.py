from functools import partial
import logging
import sys
import time

import gym
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Dense


class VanillaPolicyGradientAgent:
    """Agent that uses vanilla policy gradient (VPG)

    This trains a policy network by directly following the gradient
    of the sample estimate of the expected episode return
    with respect to the parameters of the policy network.

    Parameters
    ----------
    policy network : tf.keras.Model
        policy network that predicts the action given the observation
    optimizer : tf.keras.optimizers.Optimizer
        The optimizer to use for optimizing the parameters in `policy_network`
    minimum_batch_size : int, optional
        The minimum amount of experience to collect with current
        parameters before doing an update. If this is reached while an
        episode is in progress, the episode will be completed.
    use_reward_to_go : bool, optional
        If True, uses the reward to weight the gradients.
        This helps to reduce variance without biasing it at all, so is recommended.
        There is no bias because because a given action cannot affect the reward
        from earlier timesteps. It reduces variance by eliminating the variance
        due to those earlier timesteps. (variances are positive and add)
        If False, uses the overall return to weight the gradients.
    """

    def __init__(
        self, policy_network, optimizer, minimum_batch_size=5000, use_reward_to_go=True
    ):
        self.policy_network = policy_network
        self.optimizer = optimizer
        self.minimum_batch_size = minimum_batch_size
        self.use_reward_to_go = use_reward_to_go

    def test(self, env):
        """Tests the agent on an environment

        Parameters
        ----------
        env : gym.Environment
            The OpenAI gym environment to use for testing
        render : bool, optional
            True to render the episode on screen

        Returns
        -------
        episode_return : float
            The return at the end of the episode.
        """
        observation = env.reset()
        finished = False
        episode_return = 0

        while not finished:
            # Sample an action using the policy
            logits = self.policy_network(observation[None, :])
            action = tf.squeeze(tf.random.categorical(logits, 1)).numpy()

            # Take that action and get results from environment
            observation, reward, finished, _ = env.step(action)

            # Accumulate the return for the episode
            episode_return += reward

            # Render the environment
            env.render()

        return episode_return

    def train(self, env, epochs=50):
        """Trains the agent on an environment

        Parameters
        ----------
        env : gym.Environment
        epochs : int
            The number of epochs to train the agent for. One epoch
            is defined as a single gradient update on a batch of training data
        """
        num_actions = env.action_space.n

        for epoch in range(epochs):
            # Reset batches
            observations = []
            actions = []
            weights = []
            episode_returns = []
            episode_lengths = []

            # Run episodes until batch is full
            while len(observations) < self.minimum_batch_size:
                # Reset the environment for a new episode
                observation = env.reset()
                finished = False
                episode_length = 0
                rewards = []

                # Take steps until the episode terminates
                while not finished:
                    observations.append(observation.copy())

                    # Sample an action using the policy
                    logits = self.policy_network(observation[None, :])
                    action = tf.squeeze(tf.random.categorical(logits, 1)).numpy()
                    actions.append(action)

                    # Take that action and get results from environment
                    observation, reward, finished, _ = env.step(action)

                    # Update return and episode length
                    rewards.append(reward)
                    episode_length += 1

                    episode_return = sum(rewards)
                if self.use_reward_to_go:
                    # Reward to go is the reverse cumulative sum of the rewards
                    weights += reversed(np.cumsum(list(reversed(rewards))))
                else:
                    weights += episode_return * episode_length

                episode_returns.append(episode_return)
                episode_lengths.append(episode_length)

            # Take a single gradient step minimizing the loss function
            _loss = partial(
                policy_gradient_loss,
                policy_network=self.policy_network,
                observations=observations,
                actions=actions,
                weights=weights,
            )
            self.optimizer.minimize(_loss, self.policy_network.trainable_variables)

            # Print metrics
            loss = _loss()
            average_return = np.mean(episode_returns)
            average_length = np.mean(episode_lengths)
            print(
                f'Epoch: {epoch}\t Loss: {loss}\t return: {average_return}\t episode_length {average_length}'
            )


def policy_gradient_loss(policy_network, observations, actions, weights):
    """Vanilla policy gradient loss

    The gradient of this loss is equal to the gradient of the expected return
    only when evaluated on data obtained with the policy

    Parameter
    ----------
    policy_network : tf.keras.Model
        The policy network to use for prediction
    observations : tf.Tensor
        Shape: (batch_size, observation_dimensions)
        A batch of state observations from running the policy
    actions : tf.Tensor
        Shape: (batch_size, 1)
        A batch of actions from running the policy
    weights : tf.Tensor
        Shape: (batch_size, 1)
        A batch of weights. In the simplest case, these are the returns obtained
        from that episode.

    Returns
    -------
    loss : tf.Tensor
    """
    logits = tf.cast(policy_network(np.array(observations)), tf.float32)
    return keras.losses.SparseCategoricalCrossentropy(from_logits=True)(
        actions, logits, sample_weight=weights
    )


def multilayer_perceptron(sizes, activation='tanh', output_activation=None):
    """Multilayer perceptron

    Parameters
    ----------
    sizes : [int]
        The sizes of each layer of the network
    activation : str | callable, optional
        The activation function or name of it to use for hidden layers
    output_activation : str | callable | None, optional
        The activation function to use for the final layer. If None,
        no activation is applied

    Returns
    -------
    tf.keras.Model
        The multilayer perceptron model
    """
    model = keras.Sequential()
    for size in sizes[:-1]:
        model.add(Dense(size, activation=activation))
    model.add(Dense(sizes[-1], activation=output_activation))
    return model


if __name__ == '__main__':
    env = gym.make('CartPole-v0')
    model = multilayer_perceptron(sizes=([32] + [env.action_space.n]))
    optimizer = keras.optimizers.Adam(learning_rate=1e-2)

    agent = VanillaPolicyGradientAgent(model, optimizer)
    for _ in range(10):
        rewards_history = agent.train(env, epochs=5)
        print('Testing...')
        test_return = agent.test(env)
        print(f'Testing received return of {test_return}')
