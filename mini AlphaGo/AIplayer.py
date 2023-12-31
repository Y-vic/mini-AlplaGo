from MonteCarlo import MonteCarlo


class AIPlayer:
    def __init__(self, color: str, model_save_path="./results/model.pth"):
        self.color = color.upper()
        self.comments = "请稍后，{}正在思考".format("黑棋(X)" if self.color == 'X' else "白棋(O)")
        self.model_save_path = model_save_path

    def get_move(self, board):
        print(self.comments)
        model = MonteCarlo(board, self.color, model_save_path=self.model_save_path, use_network=False)
        action = model.search()
        return action
