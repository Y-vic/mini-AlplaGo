import math
import os.path
import random
from copy import deepcopy
import csv
import torch
from func_timeout import func_timeout, FunctionTimedOut
import os.path


class Node:
    coefficient = 2

    def __init__(self, board, color, root_color, parent=None, pre_action=None):
        self.board = board
        self.color = color.upper()
        self.root_color = root_color
        self.parent = parent
        self.children = []
        self.best_child = None
        self.get_best_child()
        self.preAction = pre_action
        self.actions = list(self.board.get_legal_actions(color=color))
        self.isOver = self.game_over()
        self.reward = {'X': 0, 'O': 0}
        self.visit_count = 0
        self.value = {'X': 1e5, 'O': 1e5}
        self.isLeaf = True
        self.best_reward_child = None
        self.get_best_reward_child()

    def game_over(self):
        b_list = list(self.board.get_legal_actions('X'))
        w_list = list(self.board.get_legal_actions('O'))
        is_over = len(b_list) == 0 and len(w_list) == 0  # 返回值 True/False
        return is_over

    def get_value(self):
        if self.visit_count == 0:
            return
        for color in ['X', 'O']:
            self.value[color] = self.reward[color] / self.visit_count + \
                                Node.coefficient * math.sqrt(
                math.log(self.parent.visit_count) / self.visit_count)  # 原本log前面有一个乘2，我对照原公式删掉了

    def add_child(self, child):
        self.children.append(child)
        self.get_best_child()
        self.get_best_reward_child()
        self.isLeaf = False

    def get_best_child(self):
        if len(self.children) == 0:
            self.best_child = None
        else:
            sorted_children = sorted(self.children, key=lambda child: child.value[self.color], reverse=True)
            self.best_child = sorted_children[0]
        return self.best_child

    def get_best_reward_child(self):
        if len(self.children) == 0:
            best_reward_child = None
        else:
            sorted_children = sorted(self.children, key=lambda child: child.reward[
                                                                          self.color] / child.visit_count if child.visit_count > 0 else -1e5,
                                     reverse=True)
            best_reward_child = sorted_children[0]
        self.best_reward_child=best_reward_child
        return self.best_reward_child


class MonteCarlo:
    def __init__(self, board, color, model_save_path, use_network):
        self.root = Node(board=deepcopy(board), color=color, root_color=color)
        self.color = color
        self.experience = {"state": [], "reward": [], "color": []}
        self.max_experience = 10000000000
        self.trans = {"X": 1, "O": -1, ".": 0}
        self.learning_rate = 0.3
        # self.epsilon的概率随机选择，每次选择后会以self.gamma的概率衰减
        self.epsilon = 0.3
        self.gamma = 0.999

        self.model = None
        self.model_save_path = model_save_path
        self.device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
        self.use_network = use_network
        self.build_network()

    def get_experience(self):
        # 广度优先搜索,添加训练数据到experience中
        # 数据量最大为self.max_experience，数据格式为棋盘局面加上value值，不包含所有叶节点
        queue = []
        for child in self.root.children:
            queue.append(child)
        while len(queue) > 0:
            if len(self.experience) == self.max_experience:
                break
            if not queue[0].isLeaf:
                self.add_one_experience(queue[0])
                for child in queue[0].children:
                    queue.append(child)
            queue.pop(0)

    def add_one_experience(self, node: Node):

        if len(self.experience["reward"]) == self.max_experience:
            return

        # 把局面转换为一个64大小的数组
        experience = self.get_state(node)
        # eg: 白棋下，棋局->黑棋的reward
        self.experience["state"].append(experience)
        reward = node.reward["X" if node.color == "O" else "O"] / node.visit_count
        self.experience["reward"].append(reward)
        self.experience["color"].append(node.color)

    def get_state(self, node):
        # state = []
        # for i in range(8):
        #     for cell in node.board._board[i]:
        #         state.append(self.trans[cell])
        state=node.board._board
        return state

    def build_network(self):
        if self.use_network:
            from train import train
            from Network import Network
            """build evaluate model"""
            self.model = Network(seed=326).to(self.device)
            if not os.path.exists(self.model_save_path):
                try:
                    time = 3600
                    func_timeout(timeout=time, func=self.build_montecarlo_tree)
                except FunctionTimedOut:
                    self.get_experience()
                    with open("./dataset/states.csv", 'w', newline='') as file:
                        writer = csv.writer(file)
                        writer.writerows(self.experience['state'])
                    with open("./dataset/values.csv", 'w', newline='') as file:
                        writer = csv.writer(file)
                        writer.writerows(self.experience['reward'])
                    # exit("finish!")
                    train(self.model_save_path)
                    print("train complete!")
            self.model.load_state_dict(torch.load(self.model_save_path))

    def search_by_network(self):
        self.model.eval()
        if self.root.isLeaf:
            self.expand(self.root)
        with torch.no_grad():
            for child in self.root.children:
                state = torch.tensor(self.get_state(child), dtype=torch.float32, device=self.device)
                value = self.model(state)
                value = value.tolist()
                child.value['X'] = value[0]
                child.value['O'] = value[1]
        return self.root.get_best_reward_child().preAction

    def search(self):
        if len(self.root.actions) == 1:
            return self.root.actions[0]
        if self.use_network:
            return self.search_by_network()
        return self.search_by_mcts()

    def search_by_mcts(self):
        try:
            func_timeout(timeout=30, func=self.build_montecarlo_tree)
            # self.build_montecarlo_tree()
        except FunctionTimedOut:
            pass

        return self.root.get_best_reward_child().preAction

    def build_montecarlo_tree(self):
        while True:
            current_node = self.select()
            if current_node.isOver:
                winner, diff = current_node.board.get_winner()
            else:
                if current_node.visit_count:
                    current_node = self.expand(current_node)
                winner, diff = self.simulation(current_node)
            self.backpropagation(node=current_node, winner=winner, diff=diff)

    def select(self):
        current_node = self.root
        while not current_node.isLeaf:
            if random.random() > self.epsilon:
                current_node = current_node.get_best_child()
            else:
                current_node = random.choice(current_node.children)
            self.epsilon *= self.gamma
        return current_node

    def simulation(self, node: Node):
        board = deepcopy(node.board)
        color = node.color
        while not self.game_over(board=board):
            actions = list(board.get_legal_actions(color=color))
            if len(actions) != 0:
                board._move(random.choice(actions), color)
            color = 'X' if color == 'O' else 'O'
        winner, diff = board.get_winner()
        return winner, diff

    def game_over(self, board):
        b_list = list(board.get_legal_actions('X'))
        w_list = list(board.get_legal_actions('O'))
        is_over = len(b_list) == 0 and len(w_list) == 0  # 返回值 True/False
        return is_over

    def expand(self, node: Node):
        if len(node.actions) == 0:
            board = deepcopy(node.board)
            color = 'X' if node.color == 'O' else 'O'
            child = Node(board=board, color=color, parent=node, pre_action="none", root_color=self.color)
            node.add_child(child)
            return node.best_child
        for action in node.actions:
            board = deepcopy(node.board)
            board._move(action=action, color=node.color)
            color = 'X' if node.color == 'O' else 'O'
            child = Node(board=board, color=color, parent=node, pre_action=action, root_color=self.color)
            node.add_child(child=child)
        return node.best_child

    def backpropagation(self, node: Node, winner, diff):
        while node is not None:
            node.visit_count += 1
            # 通过get_winner得到的winner是0代表黑棋赢
            if winner == 0:
                node.reward['X'] += diff
                node.reward['O'] -= diff
            elif winner == 1:
                node.reward['O'] += diff
                node.reward['X'] -= diff
            elif winner == 2:
                pass
            # node.reward[winner]+=diff
            # node.reward['O' if winner==0 else 'X']-=diff
            if node is not self.root:
                node.parent.visit_count += 1
                for child in node.parent.children:
                    child.get_value()
                node.parent.visit_count -= 1
            node = node.parent
