import gym

# from gym.wrappers.monitoring.video_recorder import VideoRecorder
import sys
from agents.ddpg.agent import DDPG

env = gym.make("BipedalWalker-v2")
# video = VideoRecorder(env, base_path="./video")

agent = DDPG(env)
episode_count = 3000

for i in range(episode_count):
    episode_reward = 0
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

        # Have agent take action
        agent.step(action, reward, next_observation, done)
        # TODO: Explore if we can just capture observation when env.steps on line 23
        observation = next_observation

        if done:
            print(
                f"Episode {i + 1} finished after {t + 1} timesteps. Reward: {episode_reward}. Final hull x: {env.hull.position.x}"
            )
            break

    sys.stdout.flush()

# video.close()
env.close()
