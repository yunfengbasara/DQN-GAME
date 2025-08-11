import gymnasium as gym
from gymnasium import spaces
import numpy as np
from typing import Optional, Tuple, Dict, Any


class TicTacToeEnv(gym.Env):
    """
    井字棋环境
    
    观察空间: 3x3的棋盘，0表示空位，1表示玩家1(X)，-1表示玩家2(O)
    动作空间: 0-8，对应棋盘的9个位置
    
    棋盘位置编号:
    0 | 1 | 2
    ---------
    3 | 4 | 5
    ---------
    6 | 7 | 8
    """
    
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 1}
    
    def __init__(self, render_mode: Optional[str] = None):
        super().__init__()
        
        # 动作空间：9个位置
        self.action_space = spaces.Discrete(9)
        
        # 观察空间：3x3的棋盘，每个位置可以是-1, 0, 1
        self.observation_space = spaces.Box(
            low=-1, high=1, shape=(3, 3), dtype=np.int8
        )
        
        self.render_mode = render_mode
        
        # 游戏状态
        self.board = None
        self.current_player = None  # 1表示玩家1(X)，-1表示玩家2(O)
        self.winner = None
        self.game_over = False
        
        # 用于渲染
        self._player_symbols = {0: " ", 1: "X", -1: "O"}
        
    def reset(self, seed: Optional[int] = None, options: Optional[Dict] = None) -> Tuple[np.ndarray, Dict]:
        """重置游戏状态"""
        super().reset(seed=seed)
        
        # 初始化空棋盘
        self.board = np.zeros((3, 3), dtype=np.int8)
        
        # 玩家1(X)先手
        self.current_player = 1
        
        # 重置游戏状态
        self.winner = None
        self.game_over = False
        
        observation = self._get_observation()
        info = self._get_info()
        
        if self.render_mode == "human":
            self.render()
            
        return observation, info
    
    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """执行一步动作"""
        if self.game_over:
            raise RuntimeError("游戏已结束，请调用reset()重新开始")
            
        if not self.action_space.contains(action):
            raise ValueError(f"无效动作: {action}")
            
        # 将动作转换为棋盘坐标
        row, col = divmod(action, 3)
        
        # 检查位置是否为空
        if self.board[row, col] != 0:
            # 无效动作，给予负奖励并结束游戏
            reward = -10.0
            self.game_over = True
            observation = self._get_observation()
            info = self._get_info()
            info["invalid_move"] = True
            return observation, reward, True, False, info
        
        # 执行动作
        self.board[row, col] = self.current_player
        
        # 检查游戏是否结束
        winner = self._check_winner()
        reward = self._get_reward(winner)
        
        if winner is not None or self._is_board_full():
            self.game_over = True
            self.winner = winner
        else:
            # 切换玩家
            self.current_player *= -1
        
        observation = self._get_observation()
        info = self._get_info()
        info["invalid_move"] = False
        
        if self.render_mode == "human":
            self.render()
            
        return observation, reward, self.game_over, False, info
    
    def render(self):
        """渲染游戏状态"""
        if self.render_mode is None:
            return
            
        print("\n当前棋盘状态:")
        print("-------------")
        for i in range(3):
            row_str = ""
            for j in range(3):
                symbol = self._player_symbols[self.board[i, j]]
                row_str += f" {symbol} "
                if j < 2:
                    row_str += "|"
            print(row_str)
            if i < 2:
                print("-----------")
        print("-------------")
        
        if self.game_over:
            if self.winner is not None:
                player_name = "玩家1(X)" if self.winner == 1 else "玩家2(O)"
                print(f"\n🎉 {player_name} 获胜!")
            else:
                print("\n🤝 平局!")
        else:
            current_name = "玩家1(X)" if self.current_player == 1 else "玩家2(O)"
            print(f"\n轮到 {current_name} 下棋")
    
    def close(self):
        """清理资源"""
        pass
    
    def _get_observation(self) -> np.ndarray:
        """获取当前观察"""
        return self.board.copy()
    
    def _get_info(self) -> Dict:
        """获取额外信息"""
        return {
            "current_player": self.current_player,
            "winner": self.winner,
            "game_over": self.game_over,
            "valid_actions": self._get_valid_actions(),
        }
    
    def _get_valid_actions(self) -> list:
        """获取所有有效动作"""
        valid_actions = []
        for i in range(3):
            for j in range(3):
                if self.board[i, j] == 0:
                    valid_actions.append(i * 3 + j)
        return valid_actions
    
    def _check_winner(self) -> Optional[int]:
        """检查是否有玩家获胜"""
        # 检查行
        for row in self.board:
            if abs(sum(row)) == 3:
                return row[0]
        
        # 检查列
        for col in range(3):
            if abs(sum(self.board[:, col])) == 3:
                return self.board[0, col]
        
        # 检查对角线
        if abs(sum([self.board[i, i] for i in range(3)])) == 3:
            return self.board[0, 0]
        
        if abs(sum([self.board[i, 2-i] for i in range(3)])) == 3:
            return self.board[0, 2]
        
        return None
    
    def _is_board_full(self) -> bool:
        """检查棋盘是否已满"""
        return not np.any(self.board == 0)
    
    def _get_reward(self, winner: Optional[int]) -> float:
        """计算奖励"""
        if winner == 1:  # 玩家1(X)获胜
            return 1.0
        elif winner == -1:  # 玩家2(O)获胜
            return 1.0
        elif self._is_board_full():  # 平局
            return 0.0
        else:  # 游戏继续
            return -0.01

if __name__ == "__main__":
    # 创建环境
    env = TicTacToeEnv(render_mode="human")
    
    print("🎮 井字棋游戏开始!")
    print("动作编号对应位置:")
    print("0 | 1 | 2")
    print("---------")
    print("3 | 4 | 5") 
    print("---------")
    print("6 | 7 | 8")
    
    # 重置环境
    observation, info = env.reset()
    
    # 游戏循环
    episode_reward = 0
    step_count = 0
    
    while not env.game_over:
        print(f"\n第 {step_count + 1} 步:")
        print(f"有效动作: {info['valid_actions']}")
        
        # 随机选择动作（在实际使用中，这里可以是智能体的决策）
        valid_actions = info['valid_actions']
        if valid_actions:
            action = np.random.choice(valid_actions)
            print(f"选择动作: {action}")
            
            # 执行动作
            observation, reward, terminated, truncated, info = env.step(action)
            episode_reward += reward
            step_count += 1
            
            if terminated or truncated:
                print(f"\n游戏结束! 总奖励: {episode_reward}")
                break
        else:
            print("没有有效动作可选")
            break
    
    env.close()

