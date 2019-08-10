# Training Bipedal Robot with Deep Reinforcement Learning

Random Actions (before)             |  TD3 (after)
:-------------------------:|:-------------------------:
![Random Walking Robot](https://github.com/treybean/Training-Bipedal-Robot-with-Deep-Reinforcement-Learning/blob/master/random_walking.gif?raw=true)  |  ![TD3 Walking Robot](https://github.com/treybean/Training-Bipedal-Robot-with-Deep-Reinforcement-Learning/blob/master/td3_walking.gif?raw=true)

In this project, I explore applying two deep reinforcement learning techniques, Deep Deterministic Policy Gradient (DDPG) and Twin Delayed Deep Deterministic Policy Gradient (TD3) to try to solve the BipedalWalker-v2 environment that is part of the Box2D environments onOpenAI’s Gym. This environment consists of a 4-joint walker robot that is given positive reward for moving, "walking", forward. You can read the final report writeup [here](report.pdf).


## How to run
In order to run this, install docker and then:

1. docker-compose build
2. docker-compose up

This will run the agent specified in main.py and output results to stdout as well as save files to the root project directory.
