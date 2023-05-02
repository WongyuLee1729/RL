# pip install 'pettingzoo [mpe]'

import numpy as np
from maddpg import MADDPG
from buffer import MultiAgentReplayBuffer
# from make_env import make_env

from pettingzoo.mpe import simple_adversary_v2


def MPE_adversary(action):
    '''
    Multi-Agent Particle Environment Adversary_v2
    
    N: number of good agents and landmarks
    max_cycles: number of frames (a step for each agent) until game terminates
    continuous_actions: Whether agent action spaces are discrete(default) or continuous 
    Agents : [adversary_0, agent_0, agent_1], total 3 agents
    action shape 5
    observation shape: (8), (10)
    observation values: (-inf, inf)
    state shape: (28,)
    state values: (-inf, inf)

    Agent observation space: [self_pos, self_vel, goal_rel_position, landmark_rel_position, other_agent_rel_positions]
    Adversary observation space: [landmark_rel_position, other_agents_rel_positions]
    Agent action space: [no_action, move_left, move_right, move_down, move_up]
    Adversary action space: [no_action, move_left, move_right, move_down, move_up]
    '''
    # add adversarial agent, 2 good agent 
    # env.reset()

    action = {'adversary_0':np.argmax(action[0]),'agent_0': np.argmax(action[1]),'agent_1':np.argmax(action[2])}
    observations, rewards, terminations, truncations, infos = parallel_env.step(action)
    return observations, rewards, terminations,infos


obs_init = {'adversary_0':np.array([ 0.03810739,  0.43264642, -1.3599287 ,  1.0193259 , -0.934842  ,-0.6528883 , -1.3392047 , -0.91950065]),
       'agent_0':np.array([ 0.9729494 ,  1.0855347 ,  0.9729494 ,  1.0855347 , -0.4250867 ,1.6722142 ,  0.934842  ,  0.6528883 , -0.40436265, -0.26661238]),
        'agent_1':  np.array([ 1.3773121 ,  1.3521471 ,  1.3773121 ,  1.3521471 , -0.02072404, 1.9388264 ,  1.3392047 ,  0.91950065,  0.40436265,  0.26661238])
        }


def obs_list_to_state_vector(observation):
    state = np.array([])
    
    for key, value in obs.items():
        state = np.concatenate([state, value])
    return state

if __name__ == '__main__':

    scenario = 'simple_adversary'

    parallel_env = simple_adversary_v2.parallel_env(N=2, max_cycles=26, continuous_actions = False)
    n_agents = 3
    parallel_env.reset()
        
    actor_dims = [8, 10, 10] 
    critic_dims = sum(actor_dims)

    n_actions = 5
    maddpg_agents = MADDPG(actor_dims, critic_dims, n_agents, n_actions, 
                           fc1=64, fc2=64,  
                           alpha=0.01, beta=0.01, scenario=scenario,
                           chkpt_dir='tmp/maddpg/') # initialize 

    memory = MultiAgentReplayBuffer(1000000, critic_dims, actor_dims, n_actions, n_agents, batch_size=1024) # initialize 

    PRINT_INTERVAL = 500
    N_GAMES = 5000
    MAX_STEPS = 25
    total_steps = 0
    score_history = []
    evaluate = False
    best_score = 0

    if evaluate:
        maddpg_agents.load_checkpoint()

    for i in range(N_GAMES):
        parallel_env.reset()
        score = 0
        done = [False]*n_agents
        episode_step = 0
        while not any(done):
            if evaluate:
                parallel_env.render()
            
            if episode_step == 0:
                actions = maddpg_agents.choose_action(list(obs_init.values()))
                obs = obs_init
            else:
                actions = maddpg_agents.choose_action(list(obs.values())) 
            
            obs_, reward, termin, infos = MPE_adversary(actions)
            state = obs_list_to_state_vector(obs)
            state_ = obs_list_to_state_vector(obs_)

            if episode_step >= MAX_STEPS:
                done = [True]*n_agents

            memory.store_transition(list(obs.values()), state, np.argmax(actions,axis=1), list(reward.values()), list(obs_.values()), state_, done)

            if total_steps % 100 == 0 and not evaluate:
                maddpg_agents.learn(memory)

            obs = obs_

            score += sum(list(reward.values()))
            total_steps += 1
            episode_step += 1

        score_history.append(score)
        avg_score = np.mean(score_history[-100:])
        if not evaluate:
            if avg_score > best_score:
                maddpg_agents.save_checkpoint()
                best_score = avg_score
        if i % PRINT_INTERVAL == 0 and i > 0:
            print('episode', i, 'average score {:.1f}'.format(avg_score))
