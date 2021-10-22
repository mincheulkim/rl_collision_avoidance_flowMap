import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class SAC():
    def __init__(
        self,
        model,     # models_sac.PolicyNetGaussian, models_sac.QNet
        n_actions,
        learning_rate = [1e-4, 2e-4],     # defualt:3e-4
        reward_decay = 0.98,
        replace_target_iter = 300,
        memory_size = 5000,
        batch_size = 64,
        tau = 0.01,   # maybe 0.005?   target smoothing coefficient(default:5e-3)
        alpha = 0.5,
        auto_entropy_tuning = True,
        criterion = nn.MSELoss()
    ):
        # initialize parameters
        self.n_actions = n_actions
        self.lr = learning_rate
        self.gamma = reward_decay
        self.memory_size = memory_size
        self.batch_size = batch_size
        self.tau = tau
        self.alpha = alpha
        self.auto_entropy_tuning = auto_entropy_tuning
        self.criterion = criterion
        self._build_net(model[0], model[1])   # models_sac.PolicyNetGaussian, models_sac.QNet
        self.init_memory()

    def _build_net(self, anet, cnet):   # anet = PolicyNetGaussian, cnet=QNet
        # Policy Network
        self.actor = anet().to(device)
        self.actor_optim = optim.Adam(self.actor.parameters(), lr=self.lr[0])
        # Evaluation Critic Network (new)
        self.critic = cnet().to(device)
        self.critic_optim = optim.Adam(self.critic.parameters(), lr=self.lr[1])
        # Target Critic Network (old)
        self.critic_target = cnet().to(device)
        self.critic_target.eval()

        if self.auto_entropy_tuning == True:
            self.target_entropy = -torch.Tensor(self.n_actions).to(device)
            self.log_alpha = torch.zeros(1, requires_grad=True, device=device)
            self.alpha_optim = optim.Adam([self.log_alpha], lr=0.0001)
    
    def save_load_model(self, op, path):
        anet_path = path + "sac_anet.pt"
        cnet_path = path + "sac_cnet.pt"
        anet_path_tot = path + "sac_anet_tot.pt"
        cnet_path_tot = path + "sac_cnet_tot.pt"
        if op == "save":
            torch.save(self.critic.state_dict(), cnet_path)
            torch.save(self.actor.state_dict(), anet_path)
            print('successfully saved')
            # For Total Save
            torch.save(self.critic, cnet_path_tot)
            torch.save(self.actor, anet_path_tot)
        elif op == "load":
            #self.critic.load_state_dict(torch.load(cnet_path, map_location=device))
            #self.critic_target.load_state_dict(torch.load(cnet_path, map_location=device))
            #self.actor.load_state_dict(torch.load(anet_path, map_location=device))
            print('successfully loaded')
            #For Total Load   https://jdjin3000.tistory.com/17
            self.critic=torch.load(cnet_path_tot, map_location=device)
            self.critic_target=torch.load(cnet_path_tot, map_location=device)
            self.actor=torch.load(anet_path_tot, map_location=device)

            print('tot loaded!')
            

    def choose_action(self, s, eval=False):
        '''
        s_ts = torch.FloatTensor(np.expand_dims(s,0)).to(device)
        if eval == False:
            action, _, _ = self.actor.sample(s_ts)
        else:
            _, _, action = self.actor.sample(s_ts)
        
        action = action.cpu().detach().numpy()[0]
        '''
        s_list = np.asarray(s[0])
        goal_list = np.asarray(s[1])
        speed_list = np.asarray(s[2])

        s_list = torch.FloatTensor(np.expand_dims(s_list, 0)).to(device)
        goal_list = torch.FloatTensor(np.expand_dims(goal_list, 0)).to(device)
        speed_list = torch.FloatTensor(np.expand_dims(speed_list, 0)).to(device)

        #print('g:',goal_list,goal_list.shape)
        
        #print('lidar:',s_list,s_list.shape)
        

        if eval == False:
            action, _, _ = self.actor.sample(s_list, goal_list, speed_list)   # action: tensor[[0.5083, 0.2180]] (1,2)
            #print('action:',action)
        else:
            _, _, action = self.actor.sample(s_list, goal_list, speed_list)
        action = action.cpu().detach().numpy()[0]   # tensor(1,2) to array(2,)   # array([-0.5083, 0.2180])
        


        return action

    def init_memory(self):
        self.memory_counter = 0
        self.memory = {"s":[], "a":[], "r":[], "sn":[], "end":[]}

    def store_transition(self, s, a, r, sn, end):
        if self.memory_counter <= self.memory_size:     # 10000 from ppo_city_dense.
            self.memory["s"].append(s)
            self.memory["a"].append(a)
            self.memory["r"].append(r)
            self.memory["sn"].append(sn)
            self.memory["end"].append(end)
        else:
            index = self.memory_counter % self.memory_size    # replace old memory
            self.memory["s"][index] = s
            self.memory["a"][index] = a
            self.memory["r"][index] = r
            self.memory["sn"][index] = sn
            self.memory["end"][index] = end
        #print(self.memory_counter, self.memory)
        #print('state:',self.memory["s"][0][1])
        self.memory_counter += 1

    def soft_update(self, TAU=0.01):
        with torch.no_grad():
            for targetParam, evalParam in zip(self.critic_target.parameters(), self.critic.parameters()):
                targetParam.copy_((1 - self.tau)*targetParam.data + self.tau*evalParam.data)

    def learn(self):
        # sample batch memory from all memory
        if self.memory_counter > self.memory_size:     # memory counter(every +1 cumulative) > 10000
            sample_index = np.random.choice(self.memory_size, size=self.batch_size)      # memory size=10000, batch_size = 128 => pick 128 random number from 0~9999 [2422 972 7063 357 ... 9855]
        else:
            sample_index = np.random.choice(self.memory_counter, size=self.batch_size)
        
        #s_batch = [self.memory["s"][index] for index in sample_index]
        s_list = [self.memory["s"][index][0] for index in sample_index]
        goal_list = [self.memory["s"][index][1] for index in sample_index]
        speed_list = [self.memory["s"][index][2] for index in sample_index]
        a_batch = [self.memory["a"][index] for index in sample_index]
        r_batch = [self.memory["r"][index] for index in sample_index]
        #sn_batch = [self.memory["sn"][index] for index in sample_index]
        s_n_list = [self.memory["sn"][index][0] for index in sample_index]
        goal_n_list = [self.memory["sn"][index][1] for index in sample_index]
        speed_n_list = [self.memory["sn"][index][2] for index in sample_index]
        end_batch = [self.memory["end"][index] for index in sample_index]

        # Construct torch tensor
        #s_ts = torch.FloatTensor(np.array(s_batch)).to(device)
        s_list = torch.FloatTensor(np.array(s_list)).to(device)
        goal_list = torch.FloatTensor(np.array(goal_list)).to(device)
        speed_list = torch.FloatTensor(np.array(speed_list)).to(device)

        a_ts = torch.FloatTensor(np.array(a_batch)).to(device)
        r_ts = torch.FloatTensor(np.array(r_batch)).to(device).view(self.batch_size, 1)

        #sn_ts = torch.FloatTensor(np.array(sn_batch)).to(device)
        s_n_list = torch.FloatTensor(np.array(s_n_list)).to(device)
        goal_n_list = torch.FloatTensor(np.array(goal_n_list)).to(device)
        speed_n_list = torch.FloatTensor(np.array(speed_n_list)).to(device)

        end_ts = torch.FloatTensor(np.array(end_batch)).to(device).view(self.batch_size, 1)   # 1:keep going, 0:end
        
        # TD-target
        with torch.no_grad():
            #a_next, logpi_next, _ = self.actor.sample(sn_ts)
            a_next, logpi_next, _ = self.actor.sample(s_n_list, goal_n_list, speed_n_list)
            #q_next_target = self.critic_target(sn_ts, a_next) - self.alpha * logpi_next
            q_next_target = self.critic_target(s_n_list, goal_n_list, speed_n_list, a_next) - self.alpha * logpi_next
            q_target = r_ts + end_ts * self.gamma * q_next_target
            #print(q_next_target,q_target)
        
        # Critic loss
        #q_eval = self.critic(s_ts, a_ts)
        q_eval = self.critic(s_list, goal_list, speed_list, a_ts)
        self.critic_loss = self.criterion(q_eval, q_target)

        self.critic_optim.zero_grad()   # reset calculated parameter's gradient as set 0(initialize) from previous epoch
        self.critic_loss.backward()
        self.critic_optim.step()

        # Actor loss
        #a_curr, logpi_curr, _ = self.actor.sample(s_ts)
        a_curr, logpi_curr, _ = self.actor.sample(s_list, goal_list, speed_list)
        #q_current = self.critic(s_ts, a_curr)
        q_current = self.critic(s_list, goal_list, speed_list, a_curr)
        self.actor_loss = ((self.alpha*logpi_curr) - q_current).mean()

        self.actor_optim.zero_grad()   
        self.actor_loss.backward()   # using loss and chain rule, calculate gradient(delta w) for each layers of model
        self.actor_optim.step()      # update model's parameter(w=w-alpha*delta w)

        self.soft_update()
        
        # Adaptive entropy adjustment
        if self.auto_entropy_tuning:
            alpha_loss = -(self.log_alpha * (logpi_curr + self.target_entropy).detach()).mean()
            self.alpha_optim.zero_grad()
            alpha_loss.backward()
            self.alpha_optim.step()
            self.alpha = float(self.log_alpha.exp().detach().cpu().numpy())
        
        return float(self.actor_loss.detach().cpu().numpy()), float(self.critic_loss.detach().cpu().numpy())   # loss_a, loss_c