import core
import gymnasium as gym
import numpy as np
import math

# Q Table for storing Q-values
estimation = {}

# learning rate for Q-value updates
learning_rate = 0.1
# gamma is the discount factor as mentioned in the previous section
gamma = 0.95
# explore is the exploration rate for the epsilon-greedy policy
EPS_START = 0.9
EPS_END = 0.05
# train_times is the number of training iterations
train_times = 10000
# Global step counter
g_steps = 0


def select_action(actions, observation, explore: bool = True) -> int:
    if explore is True:
        explore_alpha = EPS_END + (EPS_START - EPS_END) * math.exp(-1. * g_steps / (train_times / 10))
        if np.random.rand() < explore_alpha:
            return np.random.choice(actions)

    state_flat = tuple(observation.flatten())
    if estimation.get(state_flat) is None:
        return np.random.choice(actions)
    else:
        q_values = estimation[state_flat]
        return np.argmax(q_values).item()

def create_game(env: gym.Env, explore: bool = True) -> list:
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

    return steps[1:]

def record_steps(steps:list, log: bool = True):
    # 反转玩家，将-1玩家视角转变成1玩家视角,神经网络统一使用1玩家视角
    for record in steps:
        if record.player == -1:
            record.player = 1
            record.state = -record.state
            record.next = -record.next

    for record in steps:
        state_flat = tuple(record.state.flatten())
        if state_flat not in estimation:
            estimation[state_flat] = np.zeros(9, dtype=np.float32)
        if record.done is False:
            next_state_flat = tuple(record.next.flatten())
            if next_state_flat not in estimation:
                estimation[next_state_flat] = np.zeros(9, dtype=np.float32)  

    if log is False:
        return

    for status in steps:
        print(f"state: {status.state}, player:{status.player} action: {status.action}, reward: {status.reward}")
        print(f"next_state: {status.next} done:{status.done}")

def train(steps:list):
    for record in steps:
        state_flat = tuple(record.state.flatten())
        next_state_flat = tuple(record.next.flatten())

        # Update Q-value using the Bellman equation
        q_value = estimation[state_flat][record.action]
        max_next_q_value = np.max(estimation[next_state_flat]) if record.done is False else 0
        new_q_value = record.reward + gamma * max_next_q_value

        estimation[state_flat][record.action] += learning_rate * (new_q_value - q_value)

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
    gym.register(
        id='TicTacToe-v0',
        entry_point='agent:TicTacToeEnv',
        max_episode_steps=100,
    )

    env_train = gym.make('TicTacToe-v0')

    for times in range(train_times):
        g_steps = times
        steps = create_game(env_train)
        record_steps(steps, False)
        train(steps)
    
    env_train.close()

    human_test()