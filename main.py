import core
import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import os
import math

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

policy_net = core.NeuralNetwork().to(device)
target_net = core.NeuralNetwork().to(device)
target_net.load_state_dict(policy_net.state_dict())

optimizer = optim.Adam(policy_net.parameters(), lr=1e-4)
loss_fn = nn.SmoothL1Loss()

replay_buffer = core.ReplayBuffer(capacity=1024)
# batch_size is the number of experiences to sample from the replay buffer
batch_size = 128
# TAU is the soft update coefficient for the target network
TAU = 0.07
# gamma is the discount factor as mentioned in the previous section
gamma = 0.95
# explore is the exploration rate for the epsilon-greedy policy
EPS_START = 0.9
EPS_END = 0.05
# train_times is the number of training iterations
train_times = 30000
# Global step counter
g_steps = 0

gym.register(
    id='TicTacToe-v0',
    entry_point='agent:TicTacToeEnv',
    max_episode_steps=100,
)

env_train = gym.make('TicTacToe-v0')
env_human = gym.make('TicTacToe-v0', render_mode='human')

def select_action(actions, observation, explore: bool = True) -> int:
    if explore is True:
        explore_alpha = EPS_END + (EPS_START - EPS_END) * math.exp(-1. * g_steps / (train_times / 10))
        if np.random.rand() < explore_alpha:
            return np.random.choice(actions)

    state_flat = observation.flatten().astype(np.float32)
    state_tensor = torch.tensor(state_flat, dtype=torch.float32, device=device).unsqueeze(0)
    
    with torch.no_grad():
        q_values = policy_net(state_tensor)
    return q_values.argmax().item()
    
def create_game(env: gym.Env, explore: bool = True) -> list:
    # 默认占位
    steps = [core.Status(None, None, None, None, None, None)]

    observation, info = env.reset()
    
    while not info['game_over']:
        valid_actions = info['valid_actions']
        curstate = observation
        player = info['current_player']

        # 将player == -1转为1视角
        action = select_action(valid_actions, player * observation, explore)

        observation, reward, gameover, _, info = env.step(action)

        currecord = core.Status(curstate, player, action, reward, gameover, None)
        steps.append(currecord)

        # 修改上个步骤的 next_state
        steps[-2].next = observation
        steps[-2].done = gameover

        if gameover is False:
            continue

        # 填补最后一个步骤next_state
        steps[-1].next = np.zeros_like(curstate)

        # 如果平局或者分出胜负,将reward传导到倒数第二个状态
        if info["invalid_move"] is False:
            steps[-2].reward = -reward

        # 结束一局
        break

    # 删除第一个默认占位元素
    return steps[1:]

def record_steps(steps, log: bool = True):
    # 反转玩家，将-1玩家视角转变成1玩家视角,神经网络统一使用1玩家视角
    for record in steps:
        if record.player == -1:
            record.player = 1
            record.state = -record.state
            record.next = -record.next

    for record in steps:
        record.state = torch.tensor([record.state.flatten().tolist()], dtype=torch.float32, device=device)
        record.next = torch.tensor([record.next.flatten().tolist()], dtype=torch.float32, device=device)
        record.action = torch.tensor([[record.action]], dtype=torch.long, device=device)
        record.reward = torch.tensor([record.reward], dtype=torch.float32, device=device)
        record.done = torch.tensor([record.done], dtype=torch.bool, device=device)
        replay_buffer.push(record)

    if log is False:
        return

    for status in list(replay_buffer.buffer)[-len(steps):]:
        print(f"state: {status.state}, player:{status.player} action: {status.action}, reward: {status.reward}")
        print(f"next_state: {status.next} done:{status.done}")

def optimize_model():
    sample = replay_buffer.sample(batch_size)
    batch = core.Status(*zip(*sample))

    state_batch = torch.cat(batch.state)
    action_batch = torch.cat(batch.action)
    reward_batch = torch.cat(batch.reward)
    next_batch = torch.cat(batch.next)
    done_batch = torch.cat(batch.done)

    # 计算当前Q值
    state_action_values = policy_net(state_batch).gather(1, action_batch)

    # 计算目标Q值（考虑是否终止）
    with torch.no_grad():
        next_state_values = torch.zeros(batch_size, device=device)  # 默认全0
        # 找出未终止的样本索引
        non_final_mask = (done_batch == False)  # 或者 ~done_batch
        non_final_next_states = next_batch[non_final_mask]  # 只处理未终止的状态

        # 计算未终止状态的max Q值
        if len(non_final_next_states) > 0:
            next_state_values[non_final_mask] = target_net(non_final_next_states).max(1)[0]

    # 计算期望Q值（带折扣因子）
    expected_state_action_values = (next_state_values * gamma) + reward_batch

    # 计算损失并反向传播
    loss = loss_fn(state_action_values, expected_state_action_values.unsqueeze(1))

    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_value_(policy_net.parameters(), 100)  # 梯度裁剪
    optimizer.step()

def train():
    optimize_model()

    target_net_state_dict = target_net.state_dict()
    policy_net_state_dict = policy_net.state_dict()
    for key in policy_net_state_dict:
        target_net_state_dict[key] = policy_net_state_dict[key]*TAU + \
                target_net_state_dict[key]*(1-TAU)
        target_net.load_state_dict(target_net_state_dict)

def human_test():
    env = gym.make('TicTacToe-v0', render_mode='human')
    
    while True:
        # 设置当前人类玩家
        human_player = input("\n设置当前玩家(输入1为X先手，2为O后手，exit退出): ")
        if human_player.lower() == 'exit':
            break
        
        try:
            human_player = int(human_player)
            if human_player not in (1, 2):
                continue
        except ValueError:
            print("无效输入！")
            continue

        player_mapping = {'1': 1, '2': -1}
        human_player = player_mapping.get(str(human_player), human_player)

        print("game start:")
        print("7|8|9")
        print("4|5|6")
        print("1|2|3")

        observation, info = env.reset()
        while not info["game_over"]:
            if info['current_player'] == human_player:
                action = int(input())
                # 做个映射 小键盘映射到输入
                key_mapping = {
                    '7': 0, '8': 1, '9': 2,
                    '4': 3, '5': 4, '6': 5,
                    '1': 6, '2': 7, '3': 8
                }
                action = key_mapping.get(str(action), action)
            else:
                valid_actions = info['valid_actions']
                player = info['current_player']
                action = select_action(valid_actions, player * observation, False)

            observation, _, _, _, info = env.step(action)

    env.close()
    print("游戏已退出")

if __name__ == '__main__':
    if os.path.exists("policy_model.pth"):
        policy_net.load_state_dict(torch.load("policy_model.pth", weights_only=True))
    if os.path.exists("target_model.pth"):
        target_net.load_state_dict(torch.load("target_model.pth", weights_only=True))

    # first init
    while len(replay_buffer.buffer) < batch_size:
        steps = create_game(env_train)
        record_steps(steps, False)

    for times in range(train_times):
        g_steps = times
        steps = create_game(env_train)
        record_steps(steps, False)
        train()

    print("train game")
    create_game(env_human, False)

    env_train.close()
    env_human.close()

    torch.save(policy_net.state_dict(), "policy_model.pth")
    torch.save(target_net.state_dict(), "target_model.pth")

    human_test()
