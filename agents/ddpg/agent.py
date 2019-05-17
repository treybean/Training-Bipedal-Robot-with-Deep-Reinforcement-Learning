import numpy as np

from agents.ddpg.actor import Actor
from agents.ddpg.critic import Critic
from agents.utils.replay_buffer import ReplayBuffer
from agents.utils.ou_noise import OUNoise
from keras.models import load_model


class DDPG:
    """Reinforcement Learning agent that learns using DDPG."""

    def __init__(self, env):
        self.env = env
        self.state_size = env.observation_space.shape[0]
        self.action_size = env.action_space.shape[0]
        self.act_limit = self.env.action_space.high[0]

        # Actor (Policy) Model
        self.actor_local = Actor(env)
        self.actor_target = Actor(env)

        # Critic (Value) Model
        self.critic_local = Critic(env)
        self.critic_target = Critic(env)

        # Initialize target model parameters with local model parameters
        self.critic_target.model.set_weights(self.critic_local.model.get_weights())
        self.actor_target.model.set_weights(self.actor_local.model.get_weights())

        # Noise process
        self.exploration_mu = 0
        self.exploration_theta = 0.15
        self.exploration_sigma = 0.2
        self.noise = OUNoise(
            self.action_size,
            self.exploration_mu,
            self.exploration_theta,
            self.exploration_sigma,
        )

        # Replay memory
        self.buffer_size = 1000000
        self.batch_size = 100
        self.memory = ReplayBuffer(self.buffer_size, self.batch_size)

        # Algorithm parameters
        self.gamma = 0.99  # discount factor
        self.tau = 0.005  # for soft update of target parameters

        print(self.actor_local.model.summary())
        print(self.critic_local.model.summary())

    def reset_episode(self, state):
        self.noise.reset()

    def get_action(self, state, test=False):
        """Returns actions for given state(s) as per current policy."""
        state = np.reshape(state, [-1, self.state_size])
        action = self.actor_local.model.predict(state)[0]

        # add some noise for exploration
        if not test:
            action += self.noise.sample()

        return np.clip(action, -self.act_limit, self.act_limit)

    def step(self, state, action, reward, next_state, done):
        pi_loss = None
        q_loss = None

        # Save experience / reward
        self.memory.add(state, action, reward, next_state, done)

        # Learn, if enough samples are available in memory
        if len(self.memory) > self.batch_size:
            experiences = self.memory.sample()
            pi_loss, q_loss = self.learn(experiences)

        return pi_loss, q_loss

    def learn(self, experiences):
        """Update policy and value parameters using given batch of experience tuples."""
        # Convert experience tuples to separate arrays for each element (states, actions, rewards, etc.)
        states = np.vstack([e.state for e in experiences if e is not None])
        actions = (
            np.array([e.action for e in experiences if e is not None])
            .astype(np.float32)
            .reshape(-1, self.action_size)
        )
        rewards = (
            np.array([e.reward for e in experiences if e is not None])
            .astype(np.float32)
            .reshape(-1, 1)
        )
        dones = (
            np.array([e.done for e in experiences if e is not None])
            .astype(np.uint8)
            .reshape(-1, 1)
        )
        next_states = np.vstack([e.next_state for e in experiences if e is not None])

        # Get predicted next-state actions and Q values from target models
        #     Q_targets_next = critic_target(next_state, actor_target(next_state))
        actions_next = self.actor_target.model.predict_on_batch(next_states)
        Q_targets_next = self.critic_target.model.predict_on_batch(
            [next_states, actions_next]
        )

        # Compute Q targets for current states and train critic model (local)
        Q_targets = rewards + self.gamma * Q_targets_next * (1 - dones)
        q_train_loss = self.critic_local.model.train_on_batch(
            x=[states, actions], y=Q_targets
        )
        # print(f"q-loss: {q_train_loss}")

        # Train actor model (local) with custom train function
        action_gradients = np.reshape(
            self.critic_local.get_action_gradients([states, actions, 0]),
            (-1, self.action_size),
        )
        pi_loss = self.actor_local.train_fn([states, action_gradients, 1])[0]

        # print(f"pi_loss: {pi_loss}")

        # Soft-update target models
        self.soft_update(self.critic_local.model, self.critic_target.model)
        self.soft_update(self.actor_local.model, self.actor_target.model)

        return pi_loss, q_train_loss

    def soft_update(self, local_model, target_model):
        """Soft update model parameters."""
        local_weights = np.array(local_model.get_weights())
        target_weights = np.array(target_model.get_weights())

        assert len(local_weights) == len(
            target_weights
        ), "Local and target model parameters must have the same size"

        new_weights = self.tau * local_weights + (1 - self.tau) * target_weights
        target_model.set_weights(new_weights)

    def save_models(self, suffix=""):
        self.actor_local.model.save(f"actor_local{suffix}.h5")
        self.actor_target.model.save(f"actor_target{suffix}.h5")
        self.critic_local.model.save(f"critic_local{suffix}.h5")
        self.critic_target.model.save(f"critic_target{suffix}.h5")

    def load_models(self, suffix=""):
        self.actor_local.model = load_model(f"actor_local{suffix}.h5")
        self.actor_target.model = load_model(f"actor_target{suffix}.h5")
        self.critic_local.model = load_model(f"critic_local{suffix}.h5")
        self.critic_target.model = load_model(f"critic_target{suffix}.h5")
