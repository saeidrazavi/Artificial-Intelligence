from Game import Game
from Player import MiniMaxPlayer, RandomPlayer, HumanPlayer

# you can play games with different players here
# we will use this commented function to mark your hw
# you must win all the games against the random player
# using depth 4 tree and you must win against the random minimax player
# when the random probability is high.

# def get_game_result(player1, player2, num_game):
#     win, lose, draw = 0, 0, 0
#     for i in range(num_game):
#         game = Game(player1, player2, graphics=True)
#         result = game.start_game()
#         if result == 1:
#             win += 1
#         elif result == 2:
#             lose += 1
#         else:
#             draw += 1
#     return win, lose, draw
        
# def mark():
#     player1 = MiniMaxPlayer(1, depth=4)
#     player2 = RandomPlayer(2)
#     player3 = MiniMaxProbPlayer(2, depth=3, prob=0.8)
#     win1, lose1, draw1 = get_game_result(player1, player2, 10)
#     win2, lose2, draw2 = get_game_result(player1, player3, 2)
#     print(f'minimax player vs random player win={win1}, lose={lose1}, draw={draw1}')
#     print(f'minimax player vs minimax prob player win={win2}, lose={lose2}, draw={draw2}')
#     if win1+win2 == 12:
#         print('you will earn the full mark from random player vs minimax')

# mark()



player1 = MiniMaxPlayer(1, depth=4)
# player2 = RandomPlayer(2)
player2 = HumanPlayer(2)
game = Game(player1, player2)
game.start_game()
