import random


class RandomPlayer:
    def __init__(self, color):
        self.color = color.upper()
        # self.comments = "请稍后，{}正在思考".format("黑棋(X)" if self.color == 'X' else "白棋(O)")

    def get_move(self, board):
        action_list = list(board.get_legal_actions(color=self.color))
        return None if len(action_list) == 0 else random.choice(action_list)


class HumanPlayer:
    def __init__(self, color):
        self.color = color.upper()
        self.comments = "请{}输入一个合法位置(eg:'D3')，或按’Q/q‘退出：".format(
            "黑棋(X)" if self.color == 'X' else "白棋(O)")

    def get_move(self, board):
        while True:
            action = input(self.comments)
            action = action.upper()
            if action in board.get_legal_actions(color=self.color) or action == 'Q':
                return action
            print("输入不合法!")
