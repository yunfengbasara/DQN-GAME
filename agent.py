import numpy as np
import random
import math

class ChessAgent():

    def __init__(self) -> None:
        # board 0:empty 1:x 2:o
        self.board = np.zeros(9, dtype=int)
        # reverse board 0:empty 1:o 2:x
        self.rboard = np.zeros(9, dtype=int)
        # 空位置
        self.empty = 9
        # 索引 1:可下位置 0:不可下位置 低九位可用位置
        self.index = 0x01FF

    def Reset(self) -> None:
        self.board.fill(0)
        self.rboard.fill(0)
        self.empty = 9
        self.index = 0x01FF
    
    def Sample(self) -> tuple:
        cnt = random.randrange(0,self.empty)
        idx = self.index
        while cnt > 0:
            idx = idx & (idx - 1)
            cnt -= 1
        t = idx & (idx - 1)
        pos = int(math.log2(t ^ idx))
        return pos
    
    def SelectMaxScorePos(self, scores) -> int:
        score = 0.0
        pos = -1
        pidx = self.index
        while pidx > 0:
            t = pidx & (pidx - 1)
            idx = int(math.log2(t ^ pidx))
            pidx = t

            tscore = scores[idx]
            if pos == -1:
                pos = idx
                score = tscore
                continue

            if score < tscore:
                pos = idx
                score = tscore

        return pos
                   
    def Turn(self, pos, role) -> bool:
        if self.board[pos] != 0:
            return False
        self.board[pos] = role
        rrole = 2 if role == 1 else 1
        self.rboard[pos] = rrole 
        self.empty -= 1
        pos = 1 << pos
        self.index &= ~pos
        return True
    
    def Check(self, pos) -> int:
        dx = [-1, 1, 0, 0, -1, 1, -1, 1]
        dy = [0, 0, -1, 1, -1, 1, 1, -1]
        r = self.board[pos]
        x, y = divmod(pos, 3)
        for i in range(0, 8, 2):
            count = 1
            tx = x + dx[i]
            ty = y + dy[i]
            while tx >= 0 and tx < 3 \
                and ty >= 0 and ty < 3:
                if (r != self.board[tx * 3 + ty]):
                    break
                tx += dx[i]
                ty += dy[i]
                count += 1

            if count == 3:
                return r
            
            tx = x + dx[i + 1]
            ty = y + dy[i + 1]
            while tx >= 0 and tx < 3 \
                and ty >= 0 and ty < 3:
                if (r != self.board[tx * 3 + ty]):
                    break
                tx += dx[i + 1]
                ty += dy[i + 1]
                count += 1

            if count == 3:
                return r
        
        if self.empty == 0:
            return 3
        
        return 0


if __name__ == '__main__':
    rule = ChessAgent()
    role = 1
    res = 0
    while res == 0:
        pos = rule.Sample()
        rule.Turn(pos, role)
        res = rule.Check(pos)
        role = 2 if role == 1 else 1 
        print(rule.board.reshape(3,3))
        print('-----')
    
