import numpy as np
import tensorflow as tf
import gym
import time
from spinup.algos.ddpg import core
from spinup.algos.ddpg.core import get_vars


class ReplayBuffer:
    """
    A simple FIFO experience replay buffer for DDPG agents.
    """

    def __init__(self, obs_dim, act_dim, size):
        self.obs1_buf = np.zeros([size, obs_dim], dtype=np.float32)
        self.obs2_buf = np.zeros([size, obs_dim], dtype=np.float32)
        self.acts_buf = np.zeros([size, act_dim], dtype=np.float32)
        self.rews_buf = np.zeros(size, dtype=np.float32)
        self.done_buf = np.zeros(size, dtype=np.float32)
        self.ptr, self.size, self.max_size = 0, 0, size

    def store(self, obs, act, rew, next_obs, done):
        self.obs1_buf[self.ptr] = obs
        self.obs2_buf[self.ptr] = next_obs
        self.acts_buf[self.ptr] = act
        self.rews_buf[self.ptr] = rew
        self.done_buf[self.ptr] = done
        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    def sample_batch(self, batch_size=32):
        idxs = np.random.randint(0, self.size, size=batch_size)
        return dict(
            obs1=self.obs1_buf[idxs],
            obs2=self.obs2_buf[idxs],
            acts=self.acts_buf[idxs],
            rews=self.rews_buf[idxs],
            done=self.done_buf[idxs],
        )


class DDPG:
    """Reinforcement Learning agent that learns using DDPG."""

    def __init__(self, env):
        self.env = env
        self.actor_critic = core.mlp_actor_critic

        self.ac_kwargs = dict()
        self.seed = 0

        self.replay_size = int(1e6)
        self.gamma = 0.99
        self.polyak = 0.995
        self.pi_lr = 1e-3
        self.q_lr = 1e-3
        self.batch_size = 100
        self.start_steps = 10000
        self.act_noise = 0.1
        self.max_ep_len = 1600
        self.logger_kwargs = dict()
        self.save_freq = 1
        self.timesteps = 0

        tf.set_random_seed(self.seed)
        np.random.seed(self.seed)

        self.obs_dim = self.env.observation_space.shape[0]
        self.act_dim = self.env.action_space.shape[0]

        # Action limit for clamping: critically, assumes all dimensions share the same bound!
        self.act_limit = self.env.action_space.high[0]

        # Share information about action space with policy architecture
        self.ac_kwargs["action_space"] = self.env.action_space

        # Inputs to computation graph
        self.x_ph, self.a_ph, self.x2_ph, self.r_ph, self.d_ph = core.placeholders(
            self.obs_dim, self.act_dim, self.obs_dim, None, None
        )

        # Main outputs from computation graph
        with tf.variable_scope("main"):
            self.pi, self.q, self.q_pi = self.actor_critic(
                self.x_ph, self.a_ph, **self.ac_kwargs
            )

        # Target networks
        with tf.variable_scope("target"):
            # Note that the action placeholder going to actor_critic here is
            # irrelevant, because we only need q_targ(s, pi_targ(s)).
            self.pi_targ, _, self.q_pi_targ = self.actor_critic(
                self.x2_ph, self.a_ph, **self.ac_kwargs
            )

        # Experience buffer
        self.replay_buffer = ReplayBuffer(
            obs_dim=self.obs_dim, act_dim=self.act_dim, size=self.replay_size
        )

        # Count variables
        self.var_counts = tuple(
            core.count_vars(scope) for scope in ["main/pi", "main/q", "main"]
        )
        print(
            "\nNumber of parameters: \t pi: %d, \t q: %d, \t total: %d\n"
            % self.var_counts
        )

        # Bellman backup for Q function
        self.backup = tf.stop_gradient(
            self.r_ph + self.gamma * (1 - self.d_ph) * self.q_pi_targ
        )

        # DDPG losses
        self.pi_loss = -tf.reduce_mean(self.q_pi)
        self.q_loss = tf.reduce_mean((self.q - self.backup) ** 2)

        # Separate train ops for pi, q
        self.pi_optimizer = tf.train.AdamOptimizer(learning_rate=self.pi_lr)
        self.q_optimizer = tf.train.AdamOptimizer(learning_rate=self.q_lr)
        self.train_pi_op = self.pi_optimizer.minimize(
            self.pi_loss, var_list=get_vars("main/pi")
        )
        self.train_q_op = self.q_optimizer.minimize(
            self.q_loss, var_list=get_vars("main/q")
        )

        # Polyak averaging for target variables
        self.target_update = tf.group(
            [
                tf.assign(v_targ, self.polyak * v_targ + (1 - self.polyak) * v_main)
                for v_main, v_targ in zip(get_vars("main"), get_vars("target"))
            ]
        )

        # Initializing targets to match main variables
        self.target_init = tf.group(
            [
                tf.assign(v_targ, v_main)
                for v_main, v_targ in zip(get_vars("main"), get_vars("target"))
            ]
        )

        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())
        self.sess.run(self.target_init)

        # Setup model saving
        # logger.setup_tf_saver(
        #     sess, inputs={"x": x_ph, "a": a_ph}, outputs={"pi": pi, "q": q}
        # )

    def reset_episode(self, state):
        self.o = state
        self.ep_len = 0

    def get_action(self, o, test=False):
        if self.timesteps > self.start_steps:
            noise_scale = 0 if test else self.act_noise
            a = self.sess.run(self.pi, feed_dict={self.x_ph: o.reshape(1, -1)})[0]
            a += noise_scale * np.random.randn(self.act_dim)
            return np.clip(a, -self.act_limit, self.act_limit)
        else:
            return self.env.action_space.sample()

    def step(self, o, a, r, o2, d):
        self.timesteps += 1
        self.ep_len += 1

        # Ignore the "done" signal if it comes from hitting the time
        # horizon (that is, when it's an artificial terminal signal
        # that isn't based on the agent's state)
        d = False if self.ep_len == self.max_ep_len else d

        # Store experience to replay buffer
        self.replay_buffer.store(o, a, r, o2, d)

        pi_losses = []
        q_losses = []

        # Todo: Check if agent is learning or not.
        if d or (self.ep_len == self.max_ep_len):
            """
            Perform all DDPG updates at the end of the trajectory,
            in accordance with tuning done by TD3 paper authors.
            """
            for _ in range(self.ep_len):
                batch = self.replay_buffer.sample_batch(self.batch_size)
                feed_dict = {
                    self.x_ph: batch["obs1"],
                    self.x2_ph: batch["obs2"],
                    self.a_ph: batch["acts"],
                    self.r_ph: batch["rews"],
                    self.d_ph: batch["done"],
                }

                # Q-learning update
                outs = self.sess.run([self.q_loss, self.q, self.train_q_op], feed_dict)
                q_losses.append(outs[0])
                # logger.store(LossQ=outs[0], QVals=outs[1])

                # Policy update
                outs = self.sess.run(
                    [self.pi_loss, self.train_pi_op, self.target_update], feed_dict
                )
                pi_losses.append(outs[0])
                # logger.store(LossPi=outs[0])

        return pi_losses, q_losses
