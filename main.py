import gym
from gym.wrappers.monitoring.video_recorder import VideoRecorder

env = gym.make("BipedalWalker-v2")
# video = VideoRecorder(env, base_path="./video")

for i_episode in range(50):
    observation = env.reset()
    for t in range(1600):
        # env.render()
        # video.capture_frame()
        # print(observation)
        action = env.action_space.sample()
        observation, reward, done, info = env.step(action)
        # print(reward)
        if done:
            print(f"Episode {i_episode + 1} finished after {t+1} timesteps")
            break

# video.close()
env.close()
