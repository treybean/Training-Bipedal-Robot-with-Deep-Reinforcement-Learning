import numpy as np
from datetime import datetime
import os
import csv
import time
import numbers
import sys

import gym

# from gym.wrappers.monitoring.video_recorder import VideoRecorder

# Import the agent
# from agents.ddpg.agent import DDPG
from agents.td3.agent import TD3

# from agents.spinningup.ddpg.ddpg import DDPG as SU_DDPG


env_name = "BipedalWalker-v2"
env = gym.make(env_name)
test_env = gym.make(env_name)

# video = VideoRecorder(env, base_path="./video")

# Specify the agent imported above
# agent = DDPG(env)
agent = TD3(env)
# agent = SU_DDPG(env)

# Create output directory to store results and saved models
now = datetime.now()
output_directory = f"./results/{type(agent).__name__}/{now.year}-{now.month}-{now.day}-{now.hour}-{now.minute}"
if not os.path.exists(output_directory):
    os.makedirs(output_directory)

# Should the agent learn?
learn = True

# Todo: Make this a command line argument
# agent.load_models()
# learn = False

episode_count = 20000


def test_agent(env, agent, n=10):
    episode_rewards = []

    for _ in range(n):
        episode_reward = 0

        observation = env.reset()

        for t in range(1600):
            action = agent.get_action(observation, test=True)
            observation, reward, done, _ = env.step(action)

            episode_reward += reward

            if done or (t == 1599):
                episode_rewards.append(episode_reward)
                break

    return episode_rewards


# Pepare output file
output_file = open(f"{output_directory}/results.csv", "w")
output = csv.writer(output_file)

output_headers = [
    "episode",
    "steps",
    "episode_reward",
    "max_reward",
    "min_reward",
    "game_over",
]

if env.hull:
    output_headers += [
        "hull_position_x",
        "hull_position_y",
        "hull_linearVelocity_x",
        "hull_linearVelocity_y",
    ]

output_headers += [
    "max_pi_loss",
    "min_pi_loss",
    "mean_pi_loss",
    "max_q1_loss",
    "min_q1_loss",
    "mean_q1_loss",
    "max_q2_loss",
    "min_q2_loss",
    "mean_q2_loss",
    "test_episode_reward_mean",
    "test_episode_reward_max",
    "test_episode_reward_min",
]

output.writerow(output_headers)

total_steps = 0
start_time = time.time()
best_episode_reward = -np.inf

for i in range(episode_count):
    episode_reward = 0
    max_reward = -np.inf
    min_reward = np.inf

    # TODO: Explore shifting this back to env.reset()
    observation = env.reset()
    agent.reset_episode(observation)

    pi_losses = []
    q1_losses = []
    q2_losses = []

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
        pi_loss, q1_loss, q2_loss = agent.step(
            observation, action, reward, next_observation, done
        )

        if pi_loss:
            if isinstance(pi_loss, numbers.Number):
                pi_losses.append(pi_loss)
            elif type(pi_loss) == list:
                pi_losses = pi_loss

        if q1_loss:
            if isinstance(q1_loss, numbers.Number):
                q1_losses.append(q1_loss)
            elif type(q1_loss) == list:
                q1_losses = q1_loss

        if q2_loss:
            if isinstance(q2_loss, numbers.Number):
                q2_losses.append(q2_loss)
            elif type(q2_loss) == list:
                q2_losses = q2_loss

        observation = next_observation

        if done:
            total_steps += t
            output_row = [i + 1, t + 1]
            best_episode_reward = max(best_episode_reward, episode_reward)

            print(
                f"({(time.time() - start_time):.2f}) Ep: {i + 1}, timesteps: {t + 1}: Reward: total: {episode_reward:.2f}, max: {max_reward:.2f}, min: {min_reward:.2f}, avg: {(episode_reward/t):.2f}. Game over: {env.game_over}",
                end=" ",
            )
            output_row += [episode_reward, max_reward, min_reward, env.game_over]

            if env.hull:
                print(
                    f"Hull: position: {{x: {env.hull.position.x}, y: {env.hull.position.y}}}, linear velocity: {{x: {env.hull.linearVelocity.x}, y: {env.hull.linearVelocity.y}}}",
                    end=" ",
                )
                output_row += [
                    env.hull.position.x,
                    env.hull.position.y,
                    env.hull.linearVelocity.x,
                    env.hull.linearVelocity.y,
                ]

            if pi_losses:
                print(
                    f"Pi loss: max: {np.max(pi_losses):.2f}, min: {np.min(pi_losses):.2f}, avg: {np.mean(pi_losses):.2f}.",
                    end=" ",
                )
                output_row += [np.max(pi_losses), np.min(pi_losses), np.mean(pi_losses)]
            else:
                output_row += ["", "", ""]

            if q1_losses:
                print(
                    f"Q1 loss: max: {np.max(q1_losses):.2f}, min: {np.min(q1_losses):.2f}, avg: {np.mean(q1_losses):.2f}.",
                    end=" ",
                )
                output_row += [np.max(q1_losses), np.min(q1_losses), np.mean(q1_losses)]
            else:
                output_row += ["", "", ""]

            if q2_losses:
                print(
                    f"Q2 loss: max: {np.max(q2_losses):.2f}, min: {np.min(q2_losses):.2f}, avg: {np.mean(q2_losses):.2f}.",
                    end=" ",
                )
                output_row += [np.max(q2_losses), np.min(q2_losses), np.mean(q2_losses)]
            else:
                output_row += ["", "", ""]

            print("")

            if episode_reward >= 200:
                test_episode_rewards = test_agent(test_env, agent)

                print(
                    f"Tested agent results: avg: {np.mean(test_episode_rewards)}, max: {np.max(test_episode_rewards)}, min: {np.min(test_episode_rewards)}"
                )
                output_row += [
                    np.mean(test_episode_rewards),
                    np.max(test_episode_rewards),
                    np.min(test_episode_rewards),
                ]

                agent.save_models(
                    path=f"{output_directory}/", suffix=f"_episode_{i + 1}"
                )

            else:
                output_row += [-1000, -1000, -1000]

            output.writerow(output_row)

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
test_env.close()
output_file.close()

