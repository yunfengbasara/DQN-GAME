import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

import random
import math
import os
from collections import namedtuple, deque

import agent

# 是否CUDA加速
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 状态信息
class Status(object):
    def __init__(self,s,a,r,n,m):
        self.state = s
        self.action = a
        self.reward = r
        self.next = n
        self.mask = m

    def __iter__(self):
        return iter((self.state, self.action, self.reward, 
                     self.next, self.mask))

# 样本池
class ReplayMemory(object):
    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity)

    def push(self, status):
        self.memory.append(status)

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)

# 神经网络
class NeuralNetwork(nn.Module):
    def __init__(self, state, action):
        super().__init__()
        self.layer1 = nn.Linear(state, 128)
        self.layer2 = nn.Linear(128, 128)
        self.layer3 = nn.Linear(128, action)

    def forward(self, x):
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        return self.layer3(x)

# BATCH_SIZE is the number of transitions sampled from the replay buffer
# GAMMA is the discount factor as mentioned in the previous section
# EPS_START is the starting value of epsilon
# EPS_END is the final value of epsilon
# EPS_DECAY controls the rate of exponential decay of epsilon, higher means a slower decay
# TAU is the update rate of the target network
# LR is the learning rate of the AdamW optimizer
# REPLAY_SIZE is the replay memory size
# EPISODES is the train times
BATCH_SIZE = 128
GAMMA = 0.99
EPS_START = 0.9
EPS_END = 0.05
EPS_DECAY = 3000
TAU = 0.005
LR = 1e-4
REPLAY_SIZE = 10000
EPISODES = 5000

# 策略网络和价值网络
# 两个网络的初始权重相同
policy_net = NeuralNetwork(9, 9).to(device)
target_net = NeuralNetwork(9, 9).to(device)
target_net.load_state_dict(policy_net.state_dict())

# 损失函数以及反向传播方式
optimizer = optim.AdamW(policy_net.parameters(), lr=LR, amsgrad=True)
loss_fn = nn.SmoothL1Loss()

# 样本池大小
g_memory = ReplayMemory(REPLAY_SIZE)

# 游戏规则
g_rule = agent.ChessAgent()

# 运行次数用于策略的选择
g_steps = 1

# 策略(随机选择或者选择最佳策略)
def select_action(state):
    global g_rule
    global g_steps
    
    sample = random.random()
    eps_threshold = EPS_END + (EPS_START - EPS_END) * \
        math.exp(-1. * g_steps / EPS_DECAY)
    g_steps += 1

    if sample <= eps_threshold:
        pos = g_rule.Sample()
        return torch.tensor([[pos]], dtype=int, device=device)
    
    with torch.no_grad():
        values = policy_net(state)
        pos = g_rule.SelectMaxScorePos(values.squeeze(0))
        return torch.tensor([[pos]], dtype=int, device=device)

# 反向传播
def optimize_model():
    if len(g_memory) < BATCH_SIZE:
        return
    status = g_memory.sample(BATCH_SIZE)
    batch = Status(*zip(*status))

    state_batch = torch.cat(batch.state)
    action_batch = torch.cat(batch.action)
    reward_batch = torch.cat(batch.reward)

    state_action_values = policy_net(state_batch).gather(1, action_batch)

    next_state_values = torch.zeros(BATCH_SIZE, device=device)
    # 游戏结束局面过滤
    non_final_mask = tuple(map(lambda s: s is not None, batch.next))
    if any(non_final_mask):
        with torch.no_grad():
            non_final_next = torch.cat([s for s in batch.next if s is not None])
            non_mask_next = torch.cat([s for s in batch.mask if s is not None])

            # Double DQN
            policy_values = policy_net(non_final_next)
            # 找到该局面下可下的位置最大值,对已经下过的位置填入一个最小值
            mask_values = policy_values.masked_fill(non_mask_next, float('-inf'))
            policy_actions = mask_values.max(1)[1].unsqueeze(1)
            
            target_values = target_net(non_final_next)
            select_values = target_values.gather(1, policy_actions).squeeze(1)

            non_final_mask = torch.tensor(non_final_mask, device=device, dtype=torch.bool)
            next_state_values[non_final_mask] = select_values

            # DQN
            # non_final_values = target_net(non_final_next)
            # 找到该局面下可下的位置最大值,对已经下过的位置填入一个最小值
            # non_mask_values = non_final_values.masked_fill(non_mask_next, float('-inf'))
            # non_final_mask = torch.tensor(non_final_mask, device=device, dtype=torch.bool)
            # next_state_values[non_final_mask] = non_mask_values.max(1)[0]

    expected_state_action_values = (next_state_values * GAMMA) + reward_batch

    loss = loss_fn(state_action_values, expected_state_action_values.unsqueeze(1))

    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_value_(policy_net.parameters(), 100)
    optimizer.step()

# 更新策略网络并同步
def update_memory():
    optimize_model()

    target_net_state_dict = target_net.state_dict()
    policy_net_state_dict = policy_net.state_dict()
    for key in policy_net_state_dict:
        target_net_state_dict[key] = policy_net_state_dict[key]*TAU + \
                target_net_state_dict[key]*(1-TAU)
        target_net.load_state_dict(target_net_state_dict)

# 训练一次对局
def train():
    # test 路径
    # TEST_PATH = [0,8]

    global g_memory
    global g_rule
    g_rule.Reset()

    # 开始角色x
    role = 1
    # 游戏状态 0:gaming 1:x win 2:o win 3:draw
    res = 0
    # x上一轮的状态
    lastCross = None
    # o上一轮的状态
    lastCircle = None

    while res == 0:
        # 本轮局面
        board = g_rule.board.copy() if role == 1 else g_rule.rboard.copy()
        state = torch.tensor(board, dtype=torch.float, device=device).unsqueeze(0)

        # 执行本轮动作
        action = select_action(state)
        pos = action.squeeze(0).item()

        # 测试路径
        # idx = 9 - g_rule.empty
        # if idx < len(TEST_PATH):
        #     pos = TEST_PATH[idx]
        # ------------

        g_rule.Turn(pos, role)
        res = g_rule.Check(pos)

        reward = torch.tensor([-0.01], dtype=torch.float, device=device)
        
        # 本轮状态
        # 'state','action','reward','next_state'
        curStatus = Status(state, action, reward, None, None)

        # 更新上一轮next_state
        # 记录本轮Status
        if role == 1:
            if lastCross is not None: 
                lastCross.next = state
                mask = list(map(lambda s: 1 if s != 0 else 0, board))
                lastCross.mask = torch.tensor(mask, dtype=torch.bool, device=device).unsqueeze(0)
                g_memory.push(lastCross)
            lastCross = curStatus
        else:
            if lastCircle is not None: 
                lastCircle.next = state
                mask = list(map(lambda s: 1 if s != 0 else 0, board))
                lastCircle.mask = torch.tensor(mask, dtype=torch.bool, device=device).unsqueeze(0)
                g_memory.push(lastCircle)
            lastCircle = curStatus

        # 交换角色
        role = 2 if role == 1 else 1

        update_memory()

    # 更新最后一轮的next_state
    # 策略表是以x为视角的
    # o赢 负分, x赢 正分, 默认平局
    crossReward = -0.1
    circleReward = -0.1
    if res == 1:
        crossReward = 1.0
        circleReward = -1.0
    elif res == 2:
        crossReward = -1.0
        circleReward = 1.0
    
    lastCross.reward = torch.tensor([crossReward], dtype=torch.float, device=device)
    lastCircle.reward = torch.tensor([circleReward], dtype=torch.float, device=device)

    g_memory.push(lastCross)
    g_memory.push(lastCircle)

    update_memory()

def createGame():
    # test 路径
    TEST_PATH = [0,8]

    global g_rule
    g_rule.Reset()

    role = 1
    res = 0
    while res == 0:
        board = g_rule.board if role == 1 else g_rule.rboard
        state = torch.tensor(board, dtype=torch.float, device=device).unsqueeze(0)

        pos = 0
        idx = 9 - g_rule.empty
        if idx < len(TEST_PATH):
            pos = TEST_PATH[idx]
        else:
            with torch.no_grad():
                values = policy_net(state)
                pos = g_rule.SelectMaxScorePos(values.squeeze(0))
                print(state.view(3,3))
                print(values.view(3,3))
                print('-----')

        g_rule.Turn(pos, role)
        res = g_rule.Check(pos)

        role = 2 if role == 1 else 1
        # print(g_rule.board.reshape(3,3))
        # print('-----')
    if res == 3:
        print('draw game')
    else:
        print(f'{res} win')

if __name__ == '__main__':
    if os.path.exists("policy_model.pth"):
        policy_net.load_state_dict(torch.load("policy_model.pth"))
        target_net.load_state_dict(torch.load("target_model.pth"))
        EPS_START = 0.1

    createGame()

    for i_episode in range(EPISODES):
        train()

    createGame()

    torch.save(policy_net.state_dict(), "policy_model.pth")
    torch.save(target_net.state_dict(), "target_model.pth")