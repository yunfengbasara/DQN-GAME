import random
import numpy as np
from collections import deque

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

class Status(object):
    def __init__(self,s,p,a,r,d,n):
        self.state = s
        self.player = p
        self.action = a
        self.reward = r
        self.done = d
        self.next = n

    def __iter__(self):
        return iter((self.state, self.player, self.action, self.reward, self.done, self.next))

class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def push(self, record: Status):
        self.buffer.append(record)
    
    def sample(self, batch_size):
        return random.sample(self.buffer, batch_size)
    
    def __len__(self):
        return len(self.buffer)
    
class NeuralNetwork(nn.Module):
    def __init__(self, input_size=9, hidden_size=128, output_size=9):
        super(NeuralNetwork, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)
    
if __name__ == "__main__":
    policy_net = NeuralNetwork()
    optimizer = optim.Adam(policy_net.parameters(), lr=5e-4)
    loss_fn = nn.MSELoss()

    # 创建测试输入
    test_input = torch.tensor([random.choice([-1, 0, 1]) for _ in range(9)], dtype=torch.float32).unsqueeze(0)
    test_output = policy_net(test_input)
    print("初始测试输入:", test_input.tolist()[0])
    print("初始输出:", test_output.tolist()[0])
    
    # 记录初始输出用于比较
    initial_outputs = test_output.clone()

    # 训练参数
    target_value = 0.5
    target_index = 2  # 第3个输出（索引为2）
    epochs = 1000

    print(f"\n开始训练，目标：第{target_index+1}个输出逼近{target_value}")
    print("-" * 50)

    for epoch in range(epochs):
        # 随机生成输入（-1, 0, 1）
        input_tensor = torch.tensor([[random.choice([-1, 0, 1]) for _ in range(9)]], dtype=torch.float32)
        
        # 前向传播
        output = policy_net(input_tensor)
        
        # 构造目标：只改变第3个输出，其他保持不变
        target = output.detach().clone()
        target[0][target_index] = target_value
        
        # 使用loss_fn计算损失
        loss = loss_fn(output, target)
        
        # 反向传播
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # 每100步打印一次进度
        if epoch % 100 == 0:
            current_value = output[0][target_index].item()
            
            # 计算其他输出的变化
            current_test_output = policy_net(test_input)
            other_changes = []
            for i in range(9):
                if i != target_index:
                    change = abs(current_test_output[0][i].item() - initial_outputs[0][i].item())
                    other_changes.append(change)
            
            max_change = max(other_changes)
            avg_change = sum(other_changes) / len(other_changes)
            
            print(f"Epoch {epoch:4d}, Loss: {loss.item():.6f}, 输出[{target_index}]: {current_value:.4f}")
            print(f"        其他输出最大变化: {max_change:.4f}, 平均变化: {avg_change:.4f}")

    print("-" * 50)
    print("训练完成！")
    
    # 最终测试
    final_output = policy_net(test_input)
    print(f"\n最终测试结果:")
    print("输入:", test_input.tolist()[0])
    print("输出:", final_output.tolist()[0])
    print(f"输出[{target_index}]: {final_output[0][target_index].item():.4f}")
    print(f"目标值: {target_value}")
    
    # 分析其他输出的变化
    print(f"\n其他输出变化分析:")
    for i in range(9):
        if i != target_index:
            change = abs(final_output[0][i].item() - initial_outputs[0][i].item())
            status = "正常" if change < 0.1 else "较大" if change < 0.2 else "异常"
            print(f"输出[{i}]: 变化 {change:.4f} ({status})")