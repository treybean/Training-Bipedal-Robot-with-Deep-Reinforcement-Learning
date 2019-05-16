import gym
import numpy as np

# from gym.wrappers.monitoring.video_recorder import VideoRecorder
import sys
from agents.ddpg.agent import DDPG
from agents.spinningup.ddpg.ddpg import DDPG as SU_DDPG
import csv
import time
import numbers

env_name = "BipedalWalker-v2"
env = gym.make(env_name)

# TODO: Add a configuration parameter and only do this if test == True
# test_env = gym.make(env_name)

# video = VideoRecorder(env, base_path="./video")

agent = DDPG(env)
# agent = SU_DDPG(env)

# Should the agent learn?
learn = True

# Todo: Make this a command line argument
# agent.load_models()
# learn = False

episode_count = 20000
total_steps = 0

output_file = open("ddpg.csv", "w")
output = csv.writer(output_file)
output.writerow(
    [
        "episode",
        "steps",
        "episode_reward",
        "max_reward",
        "min_reward",
        "max_pi_loss",
        "min_pi_loss",
        "mean_pi_loss",
        "max_q_loss",
        "min_q_loss",
        "mean_q_loss",
    ]
)

start_time = time.time()

for i in range(episode_count):
    episode_reward = 0
    max_reward = -np.inf
    min_reward = np.inf

    # TODO: Explore shifting this back to env.reset()
    observation = env.reset()
    agent.reset_episode(observation)

    pi_losses = []
    q_losses = []

    for t in range(1600):
        # Render
        # env.render()
        # video.capture_frame()

        # Determine next action and reward based on state
        action = agent.get_action(observation)

        # Advance the environment by performing the actioin
        next_observation, reward, done, info = env.step(action)

        # Capture reward
        episode_reward += reward

        # Update min/max reward trackers
        max_reward = max(max_reward, reward)
        min_reward = min(min_reward, reward)

        # Have agent take action
        pi_loss, q_loss = agent.step(
            observation, action, reward, next_observation, done
        )

        if pi_loss:
            if isinstance(pi_loss, numbers.Number):
                pi_losses.append(pi_loss)
            elif type(pi_loss) == list:
                pi_losses = pi_loss

        if q_loss:
            if isinstance(q_loss, numbers.Number):
                q_losses.append(q_loss)
            elif type(q_loss) == list:
                q_losses = q_loss

        observation = next_observation

        if done:
            total_steps += t

            print(
                f"({(time.time() - start_time):.2f}) Ep: {i + 1}, timesteps: {t + 1}: Reward: total: {episode_reward:.2f}, max: {max_reward:.2f}, min: {min_reward:.2f}, avg: {(episode_reward/t):.2f}.",
                end=" ",
            )

            if pi_losses:
                print(
                    f"Pi loss: max: {np.max(pi_losses):.2f}, min: {np.min(pi_losses):.2f}, avg: {np.mean(pi_losses):.2f}.",
                    end=" ",
                )

            if q_losses:
                print(
                    f"Q loss: max: {np.max(q_losses):.2f}, min: {np.min(q_losses):.2f}, avg: {np.mean(q_losses):.2f}."
                )

            output.writerow(
                [
                    i + 1,
                    t + 1,
                    episode_reward,
                    max_reward,
                    min_reward,
                    np.max(pi_losses),
                    np.min(pi_losses),
                    np.mean(pi_losses),
                    np.max(q_losses),
                    np.min(q_losses),
                    np.mean(q_losses),
                ]
            )

            # if episode_reward >= 300:
            #     agent.save_models(suffix=f"_episode_{i}")

            # Every 100 episodes, report on average time/timestep and flush output_file
            if (i + 1) % 100 == 0:
                print(
                    f"Average time/timestep: {((time.time() - start_time)/total_steps):.2f}"
                )
                output_file.flush()

            break

    sys.stdout.flush()

# Save models
# agent.save_models()

# video.close()
env.close()
output_file.close()
