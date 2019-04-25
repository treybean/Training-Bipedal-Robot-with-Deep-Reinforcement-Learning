import gym
import numpy as np

# from gym.wrappers.monitoring.video_recorder import VideoRecorder
import sys
from agents.ddpg.agent import DDPG
import csv

env = gym.make("BipedalWalker-v2")
# video = VideoRecorder(env, base_path="./video")

agent = DDPG(env)
episode_count = 3000

output_file = open("ddpg.csv", "w")
output = csv.writer(output_file)
output.writerow(
    ["episode", "steps", "episode_reward", "max_reward", "min_reward", "final_hull_x"]
)

for i in range(episode_count):
    episode_reward = 0
    max_reward = -np.inf
    min_reward = np.inf

    # TODO: Explore shifting this back to env.reset()
    observation = agent.reset_episode()

    for t in range(1600):
        # Render
        # env.render()
        # video.capture_frame()

        # Determine next action and reward based on state
        action = agent.act(observation)
        next_observation, reward, done, info = env.step(action)

        # Capture reward
        episode_reward += reward

        # Update min/max reward trackers
        max_reward = max(max_reward, reward)
        min_reward = min(min_reward, reward)

        # Have agent take action
        agent.step(action, reward, next_observation, done)
        # TODO: Explore if we can just capture observation when env.steps on line 23
        observation = next_observation

        if done:
            print(
                f"Episode {i + 1} finished after {t + 1} timesteps. Reward: total: {episode_reward}, max: {max_reward}, min: {min_reward}. Final hull x: {env.hull.position.x}"
            )
            output.writerow(
                [
                    i + 1,
                    t + 1,
                    episode_reward,
                    max_reward,
                    min_reward,
                    env.hull.position.x,
                ]
            )
            output_file.flush()
            break

    sys.stdout.flush()

# video.close()
env.close()
output_file.close()
