'''
This file is used to train our machine learning model
for our chess game

It will allow the model to make a move and then the user
to input what the reqrd for the move was
'''

## input the chess playing ability

# add a function that will find all the available moves for all pieces of a color



# set up and train the model

# add functinality to input what the reward of a move was

## State for RL and input to stockfish
# who's turn it is (w/b) - khalil
# can castle? binary both white and black - khalil
# count number of half moves since pawn push/ capture?
# count of total moves 0.5 per move
# which piece can en passant - done

## All possible moves per color/turn, used as actions for RL
# Functtionality to make a move for RL
# Functionality to determine reward via stockfish
# class for bot (train and make move)

## Checkmate - Khalil
# select side feature
    # choose if white or black, choose if playing bot, 2 bots? pvp?
        # do this to train, don't need googoogaga human
# stress test


# Imports
import pygame
import sys
import pandas as pd



# Screen dimensions
WIDTH = 800
HEIGHT = 800

# Board dimensions
ROWS = 8
COLS = 8
TILE = WIDTH // COLS

class Chess_App:

    def __init__(self):
        pygame.init()
        self.screen = pygame.display.set_mode((WIDTH, HEIGHT))
        pygame.display.set_caption("Carter and Khalil's Cool Chess")
        self.game = Game()

    def run(self):
        
        screen = self.screen
        game = self.game
        drag = self.game.drag
        board = self.game.board

        while True:
            # show the board and pieces
            game.show_board(screen)
            game.highlight_moves(screen)
            game.show_pieces(screen)

            if drag.dragging:
                drag.update_blit(screen)

            # quit application if user clicks the X
            for event in pygame.event.get():
                
                # allows for clicks
                if event.type == pygame.MOUSEBUTTONDOWN:
                    drag.update_mouse(event.pos)
                    # check to see if the position of the mouse is on a piece
                    clicked_row = drag.mouse_y // TILE
                    clicked_col = drag.mouse_x // TILE

                    if board.tiles[clicked_row][clicked_col].piece_present:
                        piece = board.tiles[clicked_row][clicked_col].piece
                        board.valid_moves(piece, clicked_row, clicked_col)
                        drag.initial_pos(event.pos)
                        drag.drag_piece(piece)
                        # stack the board, highlights and pieces
                        game.show_board(screen)
                        game.highlight_moves(screen)
                        game.show_pieces(screen)


                # allows for dragging, refreshes piece and background
                # when mouse is moved
                elif event.type == pygame.MOUSEMOTION:
                    if drag.dragging:
                        drag.update_mouse(event.pos)
                        # stack the board, highlights and pieces
                        game.show_board(screen)
                        game.highlight_moves(screen)
                        game.show_pieces(screen)   
                        drag.update_blit(screen)

                # allows for dropping a piece
                elif event.type == pygame.MOUSEBUTTONUP:
                    if drag.dragging:
                        drag.update_mouse(event.pos)
                        chosen_row = drag.mouse_y // TILE
                        chosen_col = drag.mouse_x // TILE

                        # check to see if the move is valid
                        initial_pos = Tile(drag.init_row, drag.init_col)
                        final_pos = Tile(chosen_row, chosen_col)
                        move = Move(initial_pos, final_pos)

                        if board.check_valid(drag.piece, move):
                            board.move_piece(drag.piece, move)
                            # stack the board and pieces
                            game.show_board(screen)
                            game.show_pieces(screen)
                            # tile = board.tiles[0][3]
                            # print(tile.piece.name)
                            # print(tile.piece.color)
                            game.get_state()
                            print(game.board.state)

                    drag.drop_piece()
                
                elif event.type == pygame.QUIT:
                    pygame.quit()
                    sys.exit()

                elif event.type == pygame.QUIT:
                    pygame.quit()
                    sys.exit()
            
            # update the display after every event
            pygame.display.update()


class Game:

    def __init__(self):
        # initialize the board using the board class
        self.board = Board()
        self.drag = Drag()

    # methods to show the board and objects

    def show_board(self, surface):
        for row in range(ROWS):
            for col in range(COLS):
                # set color of tile
                # if row + col is even, color is white
                # if row + col is odd, color is black
                if (row + col) % 2 == 0:
                    color = (247, 207, 164) # light brown
                else:
                    color = (199, 141, 83) # dark brown
             
                    # draw square
                rect = (col * TILE, row * TILE, TILE, TILE)
                pygame.draw.rect(surface, color, rect)

    def show_pieces(self, surface):
        for row in range(ROWS):
            for col in range(COLS):
                # check to see if the tile has a piece
                if self.board.tiles[row][col].piece_present():
                    # piece is the piece object located on a tile on the board
                    piece = self.board.tiles[row][col].piece
                    
                    # don't show pieces that are being dragged
                    if piece != self.drag.piece:
                        # grab the image file name from the piece object
                        piece.set_img()
                        img = pygame.image.load(piece.img)
                        # set the center of the image to the center of the tile
                        img_center = col * TILE + TILE // 2, row * TILE + TILE // 2
                        piece.img_rect = img.get_rect(center=img_center)
                        # draw the image on the surface
                        surface.blit(img, piece.img_rect)

    def highlight_moves(self, surface):
        if self.drag.dragging:
            piece = self.drag.piece
            for move in piece.moves:
                # need to make a color, rect and draw the rect
                if (move.final_tile.row + move.final_tile.col) % 2 == 0:
                    color = (163, 209, 255) # light blue
                else:
                    color = (0, 122, 255) # dark blue
                rect = (move.final_tile.col * TILE, move.final_tile.row * TILE, TILE, TILE)
                pygame.draw.rect(surface, color, rect)

    def get_state(self):
        for row, _ in enumerate(self.board.tiles):
            num_open = 0
            for col, tile in enumerate(self.board.tiles[row]):
                if tile.piece_present():
                    if num_open != 0:
                        self.board.state += str(num_open)
                        num_open = 0
                        self.board.state += tile.piece.FEN_notation
                    else:
                        self.board.state += tile.piece.FEN_notation
                else:
                    num_open += 1
            if num_open != 0:
                self.board.state += str(num_open)
            if row < len(self.board.tiles)-1:
                self.board.state += '/'

        self.board.state += ' '

        # if self.board.player_color == 'white':
        #     self.board.state += 'w'
        # else:
        #     self.board.state += 'b'


# create the super class for pieces
# pieces have name and colour and an image
class Piece:

    def __init__(self, name, color, img=None, img_rect=None):
        self.name = name
        self.color = color
        self.moves = []
        self.moved = False
        self.img = img
        self.set_img()
        self.img_rect = img_rect

    def set_img(self):
        self.img = f'{self.color[0]}{self.name}.png'

    def add_move(self, move):
        self.moves.append(move)

    def clear_moves(self):
        self.moves = []

# create the children of the piece class, one for each piece
class Pawn(Piece):

    def __init__(self, color):
        # pawns only have one direction
        # white starts at the bottom and goes up (negative direction)
        # black starts at the top and goes down (positive direction)
        self.dir = -1 if color == 'white' else 1
        super().__init__('pawn', color)
        if color == "white":
            self.FEN_notation = "P"
        else:
            self.FEN_notation = "p"


class Knight(Piece):

    def __init__(self, color):
        super().__init__('knight', color)
        if color == "white":
            self.FEN_notation = "N"
        else:
            self.FEN_notation = "n"


class Bishop(Piece):

    def __init__(self, color):
        super().__init__('bishop', color)
        if color == "white":
            self.FEN_notation = "B"
        else:
            self.FEN_notation = "b"


class Rook(Piece):

    def __init__(self, color):
        super().__init__('rook', color)
        if color == "white":
            self.FEN_notation = "R"
        else:
            self.FEN_notation = "r"


class Queen(Piece):

    def __init__(self, color):
        super().__init__('queen', color)
        if color == "white":
            self.FEN_notation = "Q"
        else:
            self.FEN_notation = "q"


class King(Piece):

    def __init__(self, color):
        super().__init__('king', color)
        if color == "white":
            self.FEN_notation = "K"
        else:
            self.FEN_notation = "k"


class Tile:

    def __init__(self, row, col, piece=None):
        self.row = row
        self.col = col
        self.piece = piece

    def __eq__(self, other):
        return self.row == other.row and self.col == other.col

    # function to check if a tile has a piece
    def piece_present(self):
        if self.piece != None:
            return True
        else:
            return False

    def empty_tile(self):
        if not self.piece_present():
            return True

    # check to see if the square is an empty or
    # contains an enemy piece
    def moveable_square(self, color):
        # check to see if no piece present
        # check to see if enemy piece present
        if not self.piece_present() or self.enemy_present(color):
            return True

    # check to see if the piece in the tile is an enemy piece
    def enemy_present(self, color):
        return self.piece_present() and self.piece.color != color
    
    # check to see if the piece in the tile is a friendly piece
    def friendly_present(self, color):
        return self.piece_present() and self.piece.color == color

# use a static method to check if a position is on the board 
    @staticmethod
    def on_board(*args):
        for arg in args:
            if arg < 0 or arg > 7:
                return False
        return True


class Board:

    def __init__(self):
        self.tiles = [[0, 0, 0, 0, 0, 0, 0, 0] for col in range(COLS)]
        self.create_board()
        self.last_move = None
        self.create_pieces('white')
        self.create_pieces('black')
        # column_names=["A","B","C","D","E","F","G","H"]
        # self.state = pd.DataFrame([[(0,0) for x in range(8)] for y in range(8)],
        #                                   columns=column_names,
        #                                   dtype=object)
        # Shift index to start at 1
        # self.state.index = self.state.index + 1

        self.state = ""


    def move_piece(self, piece, move):
        initial = move.init_tile
        final = move.final_tile

        # update the board, remove the initial piece
        self.tiles[initial.row][initial.col].piece = None
        # add the piece to the final tile
        self.tiles[final.row][final.col].piece = piece

        # update that the piece has moved
        piece.moved = True

        # remove the valid moves
        piece.clear_moves()
        
        # add the move to the last move
        self.last_move = move


    def check_valid(self, piece, move):
        return move in piece.moves


    def valid_moves(self, piece, row, col):
        # determines what moves are valid for a piece
        # returns a list of valid moves
        # add carter's code for legal moves

        def iterate_moves(directions):
            for direction in directions:
                    row_dir, col_dir = direction
                    pos_move_row = row + row_dir
                    pos_move_col = col + col_dir

                    while True:
                        if Tile.on_board(pos_move_row, pos_move_col):
                            
                            # get the initial and chosen tiles
                            initial = Tile(row, col)
                            chosen = Tile(pos_move_row, pos_move_col)
                            move = Move(initial, chosen)

                            # check to see if the tile is empty
                            # continue iterating for blank tiles
                            if self.tiles[pos_move_row][pos_move_col].empty_tile():
                                piece.add_move(move)
                            
                            # check to see if the tile has an enemy piece
                            # break after adding move
                            elif self.tiles[pos_move_row][pos_move_col].enemy_present(piece.color):
                                piece.add_move(move)
                                break

                            # check to see if the tile has a friendly piece
                            elif self.tiles[pos_move_row][pos_move_col].friendly_present(piece.color):
                                break
                        
                        # break if not on board
                        else:
                            break
                        
                        # increment the position
                        pos_move_row += row_dir
                        pos_move_col += col_dir

        if isinstance(piece, Pawn):
            # check if the pawn has moved
            # BUG with diagonal moves
            if piece.moved == False:
                # pawn can move forward one square or attack diagonally
                # check to see if the tile is empty
                # check the square ahead of the pawn
                if self.tiles[row + piece.dir][col].empty_tile():
                    # create a move object
                    # check to see if on board
                    if Tile.on_board(row + piece.dir, col):
                        move = Move(Tile(row, col, piece), Tile(row + piece.dir, col))
                        piece.add_move(move)
                    # check to see if the tile two squares ahead is empty
                    if Tile.on_board(row + 2 * piece.dir, col):
                        if self.tiles[row + 2 * piece.dir][col].empty_tile():
                        # check to see if on board
                            move = Move(Tile(row, col, piece), Tile(row + 2 * piece.dir, col))
                            piece.add_move(move)
                # check the diagonal squares for an enemy piece
                # if Tile.on_board(row + piece.dir, col + 1):
                #     if self.tiles[row + piece.dir][col + 1].enemy_present(piece.color):
                #         move = Move(Tile(row, col, piece), Tile(row + piece.dir, col + 1))
                #         piece.add_move(move)
                # if Tile.on_board(row + piece.dir, col - 1):
                #     if self.tiles[row + piece.dir][col - 1].enemy_present(piece.color):
                #         move = Move(Tile(row, col, piece), Tile(row + piece.dir, col - 1))
                #         piece.add_move(move)
            # if pawn has moved, remove the two square move
            else:
                if self.tiles[row + piece.dir][col].empty_tile():
                    # check to see if on board
                    if Tile.on_board(row + piece.dir, col):
                        move = Move(Tile(row, col, piece), Tile(row + piece.dir, col))
                        piece.add_move(move)
                # if self.tiles[row + piece.dir][col + 1].enemy_present(piece.color):
                #     # check to see if on board
                #     if Tile.on_board(row + piece.dir, col + 1):
                #         move = Move(Tile(row, col, piece), Tile(row + piece.dir, col + 1))
                #         piece.add_move(move)
                # if self.tiles[row + piece.dir][col - 1].enemy_present(piece.color):
                #     if Tile.on_board[row + piece.dir][col - 1]:
                #         move = Move(Tile(row, col, piece), Tile(row + piece.dir, col - 1))
                #         piece.add_move(move)
                # these pawns can also en passant
                # enemy pawn needs to have just moved two squares to a position
                # where it is adjacent to the current pawn
                # remember to write code for en passant
                # remember to write code for promotion
                    

        elif isinstance(piece, Knight):
            knight_moves = [
                (row + 2, col + 1),
                (row + 2, col - 1),
                (row - 2, col + 1),
                (row - 2, col - 1),
                (row + 1, col + 2),
                (row + 1, col - 2),
                (row - 1, col + 2),
                (row - 1, col - 2)
            ]
            for move in knight_moves:
                pos_move_row, pos_move_col = move
                
                if Tile.on_board(pos_move_row, pos_move_col):
                    # check to see if the tile is empty or has an enemy piece
                    if self.tiles[pos_move_row][pos_move_col].moveable_square(piece.color):
                        # find the initial tile and the chosen tile
                        initial = Tile(row, col)
                        chosen = Tile(pos_move_row, pos_move_col)
                        # make a move object
                        move = Move(initial, chosen)
                        # append the move to the piece's list of moves
                        piece.add_move(move)


        elif isinstance(piece, Bishop):
            # bishop directions
            bishop_dirs = [(1, 1), 
                        (1, -1), 
                        (-1, 1), 
                        (-1, -1)
            ]
            iterate_moves(bishop_dirs)


        elif isinstance(piece, Rook):
            rook_dirs = [(1, 0),
                        (-1, 0),
                        (0, 1),
                        (0, -1)
            ]
            iterate_moves(rook_dirs)


        elif isinstance(piece, Queen):
            queen_dirs = [(1, 0),
                        (-1, 0),
                        (0, 1),
                        (0, -1),
                        (1, 1), 
                        (1, -1), 
                        (-1, 1), 
                        (-1, -1)
            ]
            iterate_moves(queen_dirs)


        elif isinstance(piece, King):
            # use the same code as the knight
            king_moves = [
                (row - 1, col - 1),
                (row + 1, col - 1),
                (row - 1, col + 1),
                (row + 1, col + 1),
                (row + 1, col),
                (row - 1, col),
                (row, col + 1),
                (row, col - 1)
            ]
            for move in king_moves:
                pos_move_row, pos_move_col = move
                
                if Tile.on_board(pos_move_row, pos_move_col):
                    # check to see if the tile is empty or has an enemy piece
                    if self.tiles[pos_move_row][pos_move_col].moveable_square(piece.color):
                        # find the initial tile and the chosen tile
                        initial = Tile(row, col)
                        chosen = Tile(pos_move_row, pos_move_col)
                        # make a move object
                        move = Move(initial, chosen)
                        # append the move to the piece's list of moves
                        piece.add_move(move)

            # need to add castling
            # need to add check and checkmate
            # need to add stalemate


    def create_board(self):
        for row in range(ROWS):
            for col in range(COLS):
                self.tiles[row][col] = Tile(row, col)

    def create_pieces(self, color):
        # white pieces will be on the front 2 rows
        # black pieces will be on the back 2 rows
        if color == 'white':
            row_pawn = 6
            row_big = 7
        else:
            row_pawn = 1
            row_big = 0

        # place pawns on the board
        for col in range(COLS):
            self.tiles[row_pawn][col] = Tile(row_pawn, col, Pawn(color))

        # place knights on the board
        self.tiles[row_big][1] = Tile(row_big, 1, Knight(color))
        self.tiles[row_big][6] = Tile(row_big, 6, Knight(color))

        # place bishops on the board
        self.tiles[row_big][2] = Tile(row_big, 2, Bishop(color))
        self.tiles[row_big][5] = Tile(row_big, 5, Bishop(color))

        # place rooks on the board
        self.tiles[row_big][0] = Tile(row_big, 0, Rook(color))
        self.tiles[row_big][7] = Tile(row_big, 7, Rook(color))

        # place queen on the board
        self.tiles[row_big][3] = Tile(row_big, 3, Queen(color))

        # place king on the board
        self.tiles[row_big][4] = Tile(row_big, 4, King(color))

        # test pieces
        # self.tiles[4][4] = Tile(4, 4, King('black'))
        # self.tiles[5][5] = Tile(5, 5, Rook('white'))
        # self.tiles[6][6] = Tile(6, 6, Queen('black'))



                    


class Drag:

    def __init__(self):
        self.mouse_x = 0
        self.mouse_y = 0
        self.init_col = 0
        self.init_row = 0
        self.piece = None
        self.dragging = False

    # function to update the image when its being dragged
    def update_blit(self, surface):
        # load the correct image
        img_path = self.piece.img
        img = pygame.image.load(img_path)
        # set the center of the image to the mouse position
        img_center = self.mouse_x, self.mouse_y
        self.piece.img_rect = img.get_rect(center=img_center)
        # blit the image
        surface.blit(img, self.piece.img_rect)


    def update_mouse(self, pos):
        # position is a tuple representing coordinates of the mouse
        self.mouse_x = pos[0]
        self.mouse_y = pos[1]

    def initial_pos(self, pos):
        self.init_col = pos[0] // TILE
        self.init_row = pos[1] // TILE

    def drag_piece(self, piece):
        self.piece = piece
        self.dragging = True

    def drop_piece(self):
        self.piece = None
        self.dragging = False


class Move:

    def __init__(self, init_tile, final_tile):
        # the arguments are tile objects
        self.init_tile = init_tile 
        self.final_tile = final_tile

    def __eq__(self, other):
        return self.init_tile == other.init_tile and self.final_tile == other.final_tile


class Bot:
    def __init__(self, model_file=None):
        self.file = model_file
        
    def train_model():
        # reset
            # Khalil made a reset function
        
        # reward
            # manual put into chess.com
            # use algorithm to calculate score of position
        # play(action)
        pass

    def save_model():
        # save the trained model
        pass

    def make_move():
        # use saved model and mave a move for the game
        pass

if __name__ == '__main__':
    chess = Chess_App()
    chess.run() 

    # import chess
    # import chess.engine

    # # Change this if stockfish is somewhere else
    # engine = chess.engine.SimpleEngine.popen_uci("C:/Users/cgaul/Desktop/CBID 2022-2023/Software Carpentry/Chess/Chess/stockfish_15.1_win_x64_avx2/stockfish-windows-2022-x86-64-avx2.exe")

    # # The position represented in FEN
    # board = chess.Board("rnbqkbnr/pp1ppppp/8/2p5/4P3/5N2/PPPP1PPP/RNBQKB1R b KQkq - 1 2")

    # # Limit our search so it doesn't run forever
    # info = engine.analyse(board, chess.engine.Limit(depth=20))

    # print("getting info")
    # print(info)
    # print("info got")
