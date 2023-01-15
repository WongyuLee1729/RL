# Python packages to use
import numpy as np
import pandas as pd
# import matplotlib.pyplot as plt
import gym
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical



env = gym.make("Taxi-v3")
env.action_space.seed(42)

#Hyperparameters
learning_rate = 0.0005
gamma = 0.98
lmbda = 0.95 # GAE에 쓰이는 계수
eps_clip = 0.1 # Clipping 범위
K_epoch = 100 # T_horizonal 만큼 쌓은 데이터를 몇번 반복할지 정함
T_horizon = 100 # update 주기 100 steps 만큼 경험을 쌓고 그 위에 policy update
episodes = 10
# s_space = np.zeros(500)

class PPO(nn.Module):
    def __init__(self):
        super(PPO, self).__init__()
        self.data = []

        self.fc1   = nn.Linear(500,512) # for taxi
        self.fc_pi = nn.Linear(512,6)
        self.fc_v  = nn.Linear(512,1)
        self.optimizer = optim.Adam(self.parameters(), lr=learning_rate)

    def pi(self, x, softmax_dim = 0):
        x = F.relu(self.fc1(x))
        x = self.fc_pi(x)
        prob = F.softmax(x, dim=softmax_dim)
        return prob
    
    def v(self, x):
        x = F.relu(self.fc1(x))
        v = self.fc_v(x)
        return v
      
    def put_data(self, transition):
        self.data.append(transition)
        
    def make_batch(self):
        s_space = np.zeros(500)
        s_prime_space = np.zeros(500)
        s_lst, a_lst, r_lst, s_prime_lst, prob_a_lst, done_lst = [], [], [], [], [], []
        for transition in self.data:
            s, a, r, s_prime, prob_a, done = transition
            s_space[s] = 1.0
            s_prime_space[s_prime] = 1.0
            s_lst.append(s_space)
            a_lst.append([a])
            r_lst.append([r])
            s_prime_lst.append(s_prime_space)
            prob_a_lst.append([prob_a])
            done_mask = 0 if done else 1
            done_lst.append([done_mask])
            s_space *= 0.0
            s_prime_space *= 0.0
        s,a,r,s_prime,done_mask, prob_a = torch.tensor(s_lst, dtype=torch.float), torch.tensor(a_lst), \
                                          torch.tensor(r_lst), torch.tensor(s_prime_lst, dtype=torch.float), \
                                          torch.tensor(done_lst, dtype=torch.float), torch.tensor(prob_a_lst)
        self.data = []
        return s, a, r, s_prime, done_mask, prob_a
        
    def train_net(self):
        s, a, r, s_prime, done_mask, prob_a = self.make_batch()

        for i in range(K_epoch):
            td_target = r + gamma * self.v(s_prime) * done_mask # s_prme = 20
            delta = td_target - self.v(s)
            delta = delta.detach().numpy()

            advantage_lst = []
            advantage = 0.0
            for delta_t in delta[::-1]:
                advantage = gamma * lmbda * advantage + delta_t[0]
                advantage_lst.append([advantage])
            advantage_lst.reverse()
            advantage = torch.tensor(advantage_lst, dtype=torch.float)

            pi = self.pi(s, softmax_dim=1)
            pi_a = pi.gather(1,a)
            ratio = torch.exp(torch.log(pi_a) - torch.log(prob_a))  # a/b == exp(log(a)-log(b))

            surr1 = ratio * advantage
            surr2 = torch.clamp(ratio, 1-eps_clip, 1+eps_clip) * advantage
            loss = -torch.min(surr1, surr2) + F.smooth_l1_loss(self.v(s) , td_target.detach())

            self.optimizer.zero_grad()
            loss.mean().backward()
            self.optimizer.step()
        

def main():
    # env = gym.make('CartPole-v1') # Taxi-v3
    env = gym.make('Taxi-v3')
    model = PPO()
    score = 0.0
    s_space = np.zeros(500)
    for n_epi in range(episodes):
        s = env.reset() 
        s_space[s]= 1.
        done = False
        while not done: # 마지막 스텝이 T_horizon 만큼 남지 않았을 경우를 위해 
            for t in range(T_horizon): # T_horizon 스텝 만큼만 data를 모으고 경험을 쌓음
                prob = model.pi(torch.from_numpy(s_space).float()) # 모델에 환경 값을 주어 확률 값을 받음
                m = Categorical(prob) # 해 당 확률을 categorial 변수로 바꾸어 줌 
                a = m.sample().item() # 샘플링을 통해 행동을 뱉음 
                s_prime, r, done, info = env.step(a) # 행동을 환경에 주어 행동에 대한 다음 값들을 받음 
                model.put_data((s, a, r, s_prime, prob[a].item(), done)) # 이전 값들을 모델에 저장해 둠 
                # prob[a].item()는 실제 내가 한 행동의 확률 값 -> PPO에서 ratio라는 값을 계산할 때 old_policy의 확률값이 쓰임
                
                s = s_prime
                s_space *= 0.0
                s_space[s]= 1.
                score += r
                if done:
                    break
            model.train_net() # T_horizon 스텝만큼 경험을 쌓은 뒤에 학습함
        if n_epi!=0:
            print("# of episode :{}, score : {:.1f}".format(n_epi, score))
            score = 0.0

    env.close()

if __name__ == '__main__':
    main()