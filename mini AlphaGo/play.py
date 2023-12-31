from game import Game
from player import HumanPlayer, RandomPlayer
from AIplayer import AIPlayer

white_player = AIPlayer('o')
#black_player = AIPlayer('x')
# white_player = AIPlayer('o', "./result/model.pth")
#white_player = RandomPlayer("o")
black_player = RandomPlayer("x")
game = Game(black_player=black_player, white_player=white_player)
game.run()
