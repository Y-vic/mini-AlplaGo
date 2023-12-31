# 黑白棋实验报告

## 题目简介

### 黑白棋

黑白棋 (Reversi)，也叫苹果棋，翻转棋，是一个经典的策略性游戏。

一般棋子双面为黑白两色，故称“黑白棋”。因为行棋之时将对方棋子翻转，则变为己方棋子，故又称“翻转棋” (Reversi) 。棋子双面为红、绿色的称为“苹果棋”。它使用 8x8 的棋盘，由两人执黑子和白子轮流下棋，最后子多方为胜方。随着网络的普及，黑白棋作为一种最适合在网上玩的棋类游戏正在逐渐流行起来。
中国主要的黑白棋游戏站点有 Yahoo 游戏、中国游戏网、联众游戏等。

黑白棋的游戏规则如下：

棋局开始时黑棋位于 E4 和 D5 ，白棋位于 D4 和 E5，如图所示。

<img src="D:\CODE\python\黑白棋\assets\image-20231231080126686.png" alt="image-20231231080126686" style="zoom:67%;" />

1. 黑方先行，双方交替下棋。
2. 一步合法的棋步包括：
   - 在一个空格处落下一个棋子，并且翻转对手一个或多个棋子；
   - 新落下的棋子必须落在可夹住对方棋子的位置上，对方被夹住的所有棋子都要翻转过来，
     可以是横着夹，竖着夹，或是斜着夹。夹住的位置上必须全部是对手的棋子，不能有空格；
   - 一步棋可以在数个（横向，纵向，对角线）方向上翻棋，任何被夹住的棋子都必须被翻转过来，棋手无权选择不去翻某个棋子。
3. 如果一方没有合法棋步，也就是说不管他下到哪里，都不能至少翻转对手的一个棋子，那他这一轮只能弃权，而由他的对手继续落子直到他有合法棋步可下。
4. 如果一方至少有一步合法棋步可下，他就必须落子，不得弃权。
5. 棋局持续下去，直到棋盘填满或者双方都无合法棋步可下。
6. 如果某一方落子时间超过 1 分钟 或者 连续落子 3 次不合法，则判该方失败。

### 实验要求

- 使用 **『蒙特卡洛树搜索算法』** 实现 miniAlphaGo for Reversi。
- 使用 Python 语言。
- 算法部分需要自己实现，不要使用现成的包、工具或者接口。

## 实验内容

### 实验基础

该实验提供Board类和Game类。

Board类主要实现棋局操作，包括棋盘的初始化（棋盘规格是 8x8，'X' 代表黑棋，'O' 代表白棋，'.' 代表未落子状态）、两类棋盘坐标之间的转化等。类(Board.py)的主要方法和属性：

- 属性：
  - `self._board`：定义当下棋盘状态

- 方法：
  - `display()`：展示棋盘
  - `board_num(action)`：棋盘坐标转化为数字坐标
  - `num_board(action)`：数字坐标转化为棋盘坐标
  - `get_legal_actions(color)`： 根据黑白棋的规则获取 color 方棋子的合法落子坐标
  - `_move(action, color)`： 根据 color 落子坐标 action 获取翻转棋子的坐标。

Game 类主要实现黑白棋的对弈，题目已经实现随机玩家和人类玩家。类（game.py）的主要方法和属性:

- 属性：
  - `self.board`：棋盘
  - `self.current_player`：定义当前的下棋一方，考虑游戏还未开始我们定义为 None
  - `self.black_player`：定义黑棋玩家 black_player
  - `self.white_player`：定义白棋玩家 white_player

- 方法：
  - `switch_player()`：下棋时切换玩家
  - `run()`：黑白棋游戏的主程序

### MonteCarlo搜索算法

**蒙特卡洛树搜索**（Monte Carlo tree search；简称：**MCTS**）是一种用于某些决策过程的启发式搜索算法。

蒙特卡洛树搜索的每个循环包括四个步骤：

- 选择（Selection）：从根节点*R*开始，连续向下选择子节点至叶子节点*L*。下文将给出一种选择子节点的方法，让游戏树向最优的方向扩展，这是蒙特卡洛树搜索的精要所在。
- 扩展（Expansion）：除非任意一方的输赢使得游戏在L结束，否则创建一个或多个子节点并选取其中一个节点*C*。
- 仿真（Simulation）：再从节点*C*开始，用随机策略进行游戏，又称为playout或者rollout。
- 反向传播（Backpropagation）：使用随机游戏的结果，更新从*C*到*R*的路径上的节点信息。

每一个节点的内容代表*胜利次数/游戏次数*

<img src="D:\CODE\python\黑白棋\assets\2560px-MCTS_(English).svg-1704015780378-1.png" alt="undefined" style="zoom:80%;" />

选择子结点的主要困难是：在较高平均胜率的移动后，在对深层次变型的利用和对少数模拟移动的探索，这二者中保持某种平衡。第一个在游戏中平衡利用与探索的公式被称为UCT（Upper Confidence Bounds to Trees，上限置信区间算法 ）UCT基于奥尔（Auer）、西萨-比安奇（Cesa-Bianchi）和费舍尔（Fischer）提出的UCB1公式，并首次由马库斯等人应用于多级决策模型（具体为马尔可夫决策过程）。科奇什和塞派什瓦里建议选择游戏树中的每个结点移动，从而使表达式$\frac{\omega_i}{n_i}+c\sqrt{\frac{lnt}{n_i}}$具有最大值。在该式中：

- $\omega_i$代表第i次移动后取胜的次数
- $n_i$代表第i次移动后仿真的次数
- c为为探索参数—理论上等于$\sqrt{2}$；在实际中通常可凭经验选择；
- t代表仿真总次数，等于所有$n_i $的和。

目前蒙特卡洛树搜索的实现大多是基于UCT的一些变形。

### MCTS实现

#### 定义节点Node类

Node类用于定义Monte Carlo tree 中的节点，该类的主要属性和方法如下：

- 属性：
  - `self.board`：当前棋局
  - `self.color`：当前下棋方
  - ` self.parent`：父节点
  - `self.children`：子节点
  - `self.reward`：节点的奖励
  - ` self.value`：根据$\frac{\omega_i}{n_i}+c\sqrt{\frac{lnt}{n_i}}$计算的节点value值，用于MCTS的selection环节
  - `self.visit_count`：节点的访问次数

- 方法：
  - `get_value()`：计算节点的value值
  - `add_child()`：添加子节点
  - `get_best_child()`：根据value值，找到节点的最好子节点
  - `get_best_reward_child()`：根据reward值，找到节点的最好子节点

具体实现见附件mini AlphaGO->MonteCarlo.py中的Node类。

#### MCTS的直接实现：MonteCarlo类

我们在MonteCarlo.py中定义MCTS的直接实现类：MonteCarlo类。该类的主要属性和方法如下：

- 属性：
  - `self.root`：根节点
  - `self.color`：根节点下棋方
  - ` self.epsilon`：selection环节中会以self.epsilon的概率随机选择子节点
  - `self.children`：self.epsilon的衰减系数

- 方法：
  - `search()`：返回MCTS得到的最佳action
  - `build_montecarlo_tree()`：建立MonteCarlo搜索树
  - `select()`：选择需要拓展的叶节点
  - `simulation()`：对初次访问的节点进行模拟，得到reward
  - `expand()`：拓展叶节点
  - `backpropagation()`：反向传播

具体实现见附件mini AlphaGO->MonteCarlo.py中的MonteCarlo类。

#### 算法具体实现流程

<img src="D:\CODE\python\黑白棋\assets\image-20231231160753255.png" alt="image-20231231160753255" style="zoom:50%;" />

输入当前的局面board和执棋方color，然后初始化MonteCarlo类后，调用search_by_mtcs()方法，指定时间后返回最优action。其中，该方法主要调用build_montecarlo_tree()，建立montecarlo_tree，并在指定时间后结束。

#### 优化

现有的MCTS算法，在进行selection时，会根据子节点的value值不断选取使得value值最大的子节点，直到到达叶节点为止。但是由于子节点在第一次建立时，会通过随机对弈的方式进行simulation从而初始化value，而一旦初始化得到的value值非常小，则在后续的迭代中，MCTS不会再选择这样的子节点进行探索，这就会导致建立的montecarlo_tree极不平衡，从而严重影响性能。为此，我们加入了随机选择策略，即在selection环节中，方法会以self.epsilon的概率随机选取一个子节点，1-self.epsilon的概率按照使子节点value值最大的策略选取。并且，因为随着程序的运行，我们所建立的montecarlo_tree越来越准确，随机探索的概率也应该逐渐下降，所以，每次选取子节点后，self.epsilon会以self.gamma的衰减因子进行衰减。

加入随机选择策略后的select()方法：

```python
    def select(self):
        current_node = self.root
        while not current_node.isLeaf:
            if random.random() > self.epsilon:
                current_node = current_node.get_best_child()
            else:
                current_node = random.choice(current_node.children)
            self.epsilon *= self.gamma
        return current_node
```

### 运行结果

不同算法实现的AI与基于MCTS的AI（本方）对局结果

| 基准                   | 本方胜场数/总场数 | 本方胜率 |
| ---------------------- | ----------------- | -------- |
| 随机策略               | 40/40             | 100%     |
| 可转化棋子最大的贪心   | 40/40             | 100%     |
| 使对手行动力最小的贪心 | 40/40             | 100%     |
| MO平台初级玩家         | 40/40             | 100%     |
| MO平台中级玩家         | 40/40             | 100%     |
| MO平台高级玩家         | 39/40             | 97.5%    |

MO平台测试结果截图（高级）：用时950s，领先28子

<img src="D:\CODE\python\黑白棋\assets\image-20231231164107166.png" alt="image-20231231164107166" style="zoom:50%;" />

## 总结

本次实验，我们采用MCTS算法实现了黑白棋AI棋手，并取得了良好的表现。在调试阶段出现的问题以及解决方法如下：

- “返回最优子节点”问题：开始时，我们在“建立montecarlo_tree”和“返回最优策略”的两种情况下均使用value值来作为最优子节点的依据，但是在运行时效果不佳。经过认真分析，我们发现在“返回最优策略”时，我们不应该再考虑未访问节点的探索问题，即对于最优子节点，我们应该采取平均reward的衡量标准。
- “MonteCarlo Tree不平衡”问题：因为子节点value值的初始化是通过随机对弈的方式进行simulation的，所以当随机结果极大或极小时，会使得模型在之后的迭代中，只选择特定几个子节点，导致搜索树的不平衡。解决方法是在select环节，加入随机选择策略，并同时使随机选择的概率随机搜索树的不断建立而降低。
- 神经网络：参考AlphaGo的思路，为了进一步提高AI棋手的效率，我们可以引入卷积神经网络和自博弈的方式更新模型，并最终保留效果最佳的模型参数。但是受限于时间、算力问题，我们的神经网络预测效果不佳，仍需进一步完善。因此，在本次实验中，我们最终并未采用神经网络，但保留了神经网络的训练以及调用接口，可参考附件mini AlphaGO->train.py, Network.py，以及mini AlphaGO->MonteCarlo.py中的build_network()等方法。

