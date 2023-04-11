#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import gym
env = gym.make("Taxi-v3")
env.action_space.seed(42)


def simple_run(env, number_of_episodes: int = 10, number_of_steps: int = 50): 
    env.reset()
    rendering = [] # for animation
    for episode in range(number_of_episodes):
        rewards = 0
        for t in range(number_of_steps): 
            action = env.action_space.sample() # automatically selects one random action from set of all possible actions.
            state, reward, done, info = env.step(action)
            rewards += reward
            rendering.append({
                    'frame': env.render(mode='ansi'),
                    'state': state,
                    'action': action,
                    'reward': reward})
            if done:
                env.reset()
                break
        print(f"score: {rewards}")
    env.close()

    def print_rendering(rendering):
        for i, frame in enumerate(rendering):
            clear_output(wait=True)
            # print(frame['frame'])
            # print(f"Timestep: {i + 1}")
            # print(f"State: {frame['state']}")
            # print(f"Action: {frame['action']}")
            # print(f"Reward: {frame['reward']}")
            sleep(.1)        
    print_rendering(rendering)
    
    return True
