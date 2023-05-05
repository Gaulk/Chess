'''
This code contains the main architecture of the chess game.
It contains the classes for the board, tiles, pieces, moves, and drag.
It also contains the main game loop.
'''

### Updates
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

import pygame
import sys
import copy
import random
import time

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

# Screen dimensions
WIDTH = 800
HEIGHT = 800

# Board dimensions
ROWS = 8
COLS = 8
TILE = WIDTH // COLS


class Chess_App:
    '''
    This class contains the main game loop
    **Attributes**
        # screen - the screen object
        # game - the game object
    **Methods**
        # __init__ - initializes the game
        # run - runs the main game loop
    '''

    def __init__(self):
        '''
        This method initializes the game
        **Parameters**
            # none
        **Returns**
            # none
        '''

        pygame.init()
        self.screen = pygame.display.set_mode((WIDTH, HEIGHT))
        pygame.display.set_caption("Carter and Khalil's Cool Chess")
        self.game = Game()


    def run(self):
        '''
        This method runs the main game loop, displaying the board and pieces
        and allowing for user input.
        **Parameters**
            # none
        **Returns**
            # none
        '''

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

                    # if clicking on an empty tile, do nothing
                    if not board.tiles[clicked_row][clicked_col].piece_present:
                        # stack the board and pieces
                        game.show_board(screen)
                        game.show_pieces(screen)
                        break

                    elif board.tiles[clicked_row][clicked_col].piece_present:
                        piece = board.tiles[clicked_row][clicked_col].piece
                        # check to see if the piece is the same color as the turn
                        if piece is not None and piece.color == game.player_color:
                            board.valid_moves(piece, clicked_row, clicked_col)
                            drag.initial_pos(event.pos)
                            drag.drag_piece(piece)
                            # stack the board, highlights and pieces
                            game.show_board(screen)
                            game.highlight_moves(screen)
                            game.show_pieces(screen)

                # allow for dragging, refresh piece and background
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
                            
                            board.legal_passant(drag.piece)

                            game.show_board(screen)
                            game.show_pieces(screen)
                            # change the turn
                            game.change_turn()
                            game.get_state()

                            game.state_tensor()
                            # print(game.board.state)
                            game.board.all_valid_moves('white')
                            # print(game.board.white_moves)
                            # model = ChessAI()
                            # output = model(game.board.tensor, len(game.board.white_moves))
                            # print(output.shape)
                            
                            # kept from khalil
                            print(game.board.state)
                            game.counter(self.game.count)

                    drag.drop_piece()        

                # if 'r' is pressed, reset the game
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_r:
                        game.reset_game()
                        game = self.game
                        board = self.game.board
                        drag = self.game.drag
                
                elif event.type == pygame.QUIT:
                    pygame.quit()
                    sys.exit()

                elif event.type == pygame.QUIT:
                    pygame.quit()
                    sys.exit()
            
            # update the display after every event
            pygame.display.update()

    def train(self):
        screen = self.screen
        game = self.game
        board = self.game.board

        while True:
            # show the board and pieces
            game.show_board(screen)
            game.highlight_moves(screen)
            game.show_pieces(screen)

            for event in pygame.event.get():
                if event.type == pygame.KEYDOWN:
                    # if 'r' is pressed, reset the game
                    if event.key == pygame.K_r:
                        game.reset_game()
                        game = self.game
                        board = self.game.board

                    # if 'n' is pressed, make a move for white and black
                    elif event.key == pygame.K_n:
                        while True:
                            board.all_valid_moves('white')
                            # print(board.white_moves)
                            move_white = random.choice(game.board.white_moves)
                            piece_white = board.tiles[move_white.init_tile.row][move_white.init_tile.col].piece
                            print(move_white.init_tile.row, move_white.init_tile.col)
                            print(move_white.final_tile.row, move_white.final_tile.col)
                            print(piece_white)
                            game.state_tensor()
                            # print(board.tensor)
                            board.move_piece(piece_white, move_white)
                            move_white.init_tile.clear_piece()
                            game.change_turn()
                            print("white move")

                            game.show_board(screen)
                            game.show_pieces(screen)
                            pygame.display.update()

                            board.all_valid_moves('black')
                            # print(board.black_moves)
                            move_black = random.choice(game.board.black_moves)
                            piece_black = board.tiles[move_black.init_tile.row][move_black.init_tile.col].piece
                            print(move_black.init_tile.row, move_black.init_tile.col)
                            print(move_black.final_tile.row, move_black.final_tile.col)
                            print(piece_black)
                            game.state_tensor()
                            # print(board.tensor)
                            board.move_piece(piece_black, move_black)
                            move_black.init_tile.clear_piece()
                            game.change_turn()
                            print("black move")

                            game.show_board(screen)
                            game.show_pieces(screen)
                            pygame.display.update()
                
                elif event.type == pygame.QUIT:
                    pygame.quit()
                    sys.exit()

            # update the display after every event
            pygame.display.update()


class Game:
    '''
    This class contains the game logic
    **Attributes**
        # board - the board object that contains the tiles
        # drag - the drag object
        # player_color - the color of the player
        # count - the number of moves
    **Methods**
        # show_board - shows the board
        # show_pieces - shows the pieces
        # highlight_moves - highlights the valid moves
        # change_turn - changes the turn
        # reset_game - resets the game
        # get_state - gets the state of the board
        # counter - counts the number of moves
    '''

    def __init__(self):
        '''
        This method initializes the game
        **Parameters**
            # none
        **Returns**
            # none
        '''

        # initialize the board using the board class
        self.board = Board()
        self.drag = Drag()
        self.ChessAI = ChessAI()
        self.player_color = 'white'
        self.count = 0


    def show_board(self, surface):
        '''
        This method shows the board
        **Parameters**
            # surface: *pygame.Surface* - the surface to draw on
        **Returns**
            # none
        '''

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
        '''
        This method shows the pieces
        **Parameters**
            # surface: *pygame.Surface*, the surface to draw on
        **Returns**
            # none
        '''

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
        '''
        This method highlights the valid moves
        **Parameters**
            # surface: *pygame.Surface*, the surface to draw on
        **Returns**
            # none
        '''

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

    def counter(self, count):
        '''
        This method counts the number of moves made
        moves increment by 0.5 for every move made
        thus after both players have made a move, the count will increment by 1
        **Parameters**
            # count: *float*, 0 at the start of the game
        **Returns**
            # count: *float*, the number of moves
        '''

        # counter for the number of moves made
        # increment the count by 0.5 for white moves
        # increment the count by 0.5 for black moves
        if self.player_color == 'white':
            self.count += 0.5
        else:
            self.count += 0.5
        return count


    def get_state(self):
        '''
        This method gets the state of the board
        **Parameters**
            # none
        **Returns**
            # none
        '''

        self.board.state = ''
        
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

        if self.player_color == 'white':
            self.board.state += 'w'
        else:
            self.board.state += 'b'

    def state_tensor(self):
        self.board.tensor = torch.zeros((6,8,8), dtype=torch.float)
        piece_dict = {
            'pawn': 0,
            'rook': 1,
            'knight': 2,
            'bishop': 3,
            'queen': 4,
            'king': 5
        }

        for row, _ in enumerate(self.board.tiles):
            for col, tile in enumerate(self.board.tiles[row]):
                if tile.piece_present():
                    if tile.piece.color == 'white':
                            self.board.tensor[piece_dict[tile.piece.name],row,col] = 1
                    elif tile.piece.color == 'black':
                            self.board.tensor[piece_dict[tile.piece.name],row,col] = -1




# method of changing tuns
    def change_turn(self):
        '''
        This method changes the turn
        **Parameters**
            # none
        **Returns**
            # none
        '''

        if self.player_color == 'black':
            self.player_color = 'white'
        else:
            self.player_color = 'black'


    def reset_game(self):
        '''
        This method resets the game
        **Parameters**
            # none
        **Returns**
            # none
        '''

        self.__init__()


class Piece:
    '''
    This is a super class that represents any piece
    **Attributes**
        # name - the name of the piece (pawn, rook, knight, bishop, queen, king)
        # color - the color of the piece (white, black)
        # moves - the valid moves the piece can make
        # moved - whether or not the piece has moved
        # img - the file name image of the piece
        # img_rect - the rectangle of the image of the piece
    **Methods**
        # __init__ - the constructor for the Piece class
        # set_img - sets the image of the piece
        # add_move - adds a move to the piece
        # clear_moves - clears the moves of the piece
    '''

    def __init__(self, name, color, img=None, img_rect=None):
        '''
        This is the constructor for the Piece class
        **Parameters**
            # name: *str*, the name of the piece (pawn, rook, knight, bishop, queen, king)
            # color: *str*, the color of the piece (white, black)
            # img: *str*, the file name image of the piece
            # img_rect: *pygame.RectValue* the rectangle of the image of the piece
        **Returns**
            # none
        '''

        self.name = name
        self.color = color
        self.moves = []
        self.moved = False
        self.img = img
        self.set_img()
        self.img_rect = img_rect


    def set_img(self):
        '''
        This method sets the image of the piece
        **Parameters**
            # none
        **Returns**
            # none
        '''

        self.img = f'{self.color[0]}{self.name}.png'


    def add_move(self, move):
        '''
        This method adds a valid move to the piece
        **Parameters**
            # move - the move to add
        **Returns**
            # none
        '''

        self.moves.append(move)


    def clear_moves(self):
        '''
        This method clears the moves of the piece
        **Parameters**
            # none
        **Returns**
            # moves: *lst*, an empty list of moves
        '''

        self.moves = []


# create the children of the piece class, one for each piece
class Pawn(Piece):
    '''
    This class represents a pawn, which is a child of the Piece class
    **Attributes**
        # dir - the direction the pawn can move
        # en_passant - whether or not the pawn can be taken en passant
        # FEN_notation - the FEN notation for the pawn
        # inherited from Piece
    **Methods**
        # none
    '''

    def __init__(self, color):
        '''
        This is the constructor for the Pawn class
        **Parameters**
            # color: *str*, the color of the pawn (white, black)
        **Returns**
            # none
        '''

        # pawns only have one direction
        # white starts at the bottom and goes up (negative direction)
        # black starts at the top and goes down (positive direction)
        self.dir = -1 if color == 'white' else 1
        self.en_passant = False
        super().__init__('pawn', color)
        if color == "white":
            self.FEN_notation = "P"
        else:
            self.FEN_notation = "p"


class Knight(Piece):
    '''
    This class represents a knight, which is a child of the Piece class
    **Attributes**
        # none, but inherits from Piece
        # FEN_notation: *str*, the FEN notation of the knight
    **Methods**
        # none
    '''

    def __init__(self, color):
        '''
        This is the constructor for the Knight class
        **Parameters**
            # color: *str*, the color of the knight (white, black)
        **Returns**
            # none
        '''

        super().__init__('knight', color)
        if color == "white":
            self.FEN_notation = "N"
        else:
            self.FEN_notation = "n"


class Bishop(Piece):
    '''
    This class represents a bishop, which is a child of the Piece class
    **Attributes**
        # none, but inherits from Piece
        # FEN_notation, *str* - the FEN notation of the bishop
    **Methods**
        # none
    '''

    def __init__(self, color):
        '''
        This is the constructor for the Bishop class
        **Parameters**
            # color: *str*, the color of the bishop (white, black)
        **Returns**
            # none
        '''

        super().__init__('bishop', color)
        if color == "white":
            self.FEN_notation = "B"
        else:
            self.FEN_notation = "b"


class Rook(Piece):
    '''
    This class represents a rook, which is a child of the Piece class
    **Attributes**
        # none, but inherits from Piece
        # FEN_notation: *str*, the FEN notation of the rook
    **Methods**
        # none
    '''

    def __init__(self, color):
        '''
        This is the constructor for the Rook class
        **Parameters**
            # color: *str*, the color of the rook (white, black)
        **Returns**
            # none
        '''

        super().__init__('rook', color)
        if color == "white":
            self.FEN_notation = "R"
        else:
            self.FEN_notation = "r"


class Queen(Piece):
    '''
    This class represents a queen, which is a child of the Piece class
    **Attributes**
        # none, but inherits from Piece
        # FEN_notation, *str* the FEN notation of the queen
    **Methods**
        # none
    '''

    def __init__(self, color):
        '''
        This is the constructor for the Queen class
        **Parameters**
            # color, *str*, the color of the queen (white, black)
        **Returns**
            # none
        '''
        super().__init__('queen', color)
        if color == "white":
            self.FEN_notation = "Q"
        else:
            self.FEN_notation = "q"


class King(Piece):
    '''
    This class represents a king, which is a child of the Piece class
    **Attributes**
        # none, but inherits from Piece
        # FEN_notation: *str*, the FEN notation of the king
    **Methods**
        # none
    '''

    def __init__(self, color):
        '''
        This is the constructor for the King class
        **Parameters**
            # color: *str*, the color of the king (white, black)
        **Returns**
            # none
        '''
        # rooks are added as attributes of the king
        # this will be used for castling
        self.l_rook = None
        self.r_rook = None
        super().__init__('king', color)
        if color == "white":
            self.FEN_notation = "K"
        else:
            self.FEN_notation = "k"


class Tile:
    '''
    This class represents a tile on the board
    **Attributes**
        # row: *int*, the row of the tile
        # col: *int*, the column of the tile
        # piece: *Piece obj* the piece on the tile
    **Methods**
        # __init__: *constructor*, creates a new tile
        # piece_present: *bool*, checks to see if a piece is present on the tile
    '''

    def __init__(self, row, col, piece=None):
        '''
        This is the constructor for the Tile class
        **Parameters**
            # row: *int*, the row of the tile
            # col: *int*, the column of the tile
            # piece: *Piece obj*, the piece on the tile
        **Returns**
            # none
        '''
        self.row = row
        self.col = col
        self.piece = piece


    def __eq__(self, other):
        '''
        Allows for the comparison of two tiles
        **Parameters**
            # other: *Tile obj* - the other tile to compare to
        **Returns**
            # *bool* - True if the tiles are equal, False otherwise
        '''

        return self.row == other.row and self.col == other.col


    def piece_present(self):
        '''
        Checks to see if a piece is present on the tile
        **Parameters**
            # none
        **Returns**
            # *bool* - True if a piece is present, False otherwise
        '''
        if self.piece != None:
            return True
        else:
            return False

    def empty_tile(self):
        '''
        Checks to see if a tile is empty
        **Parameters**
            # none
        **Returns**
            # *bool* - True if the tile is empty, False otherwise
        '''
        if not self.piece_present():
            return True
        
    def clear_piece(self):
        self.piece = None


    def moveable_square(self, color):
        '''
        Checks to see if a square is either empty or has an enemy piece
        **Parameters**
            # color: *str*, the color of the piece that is moving
        **Returns**
            # *bool* - True if the square is empty or has an enemy piece, False otherwise
        '''

        if not self.piece_present() or self.enemy_present(color):
            return True


    def enemy_present(self, color):
        '''
        Checks to see if an enemy piece is present on the tile
        **Parameters**
            # color: *str*, the color of the piece that is moving
        **Returns**
            # *bool* - True if an enemy piece is present, False otherwise
        '''
        return self.piece_present() and self.piece.color != color
    

    def friendly_present(self, color):
        '''
        Checks to see if a friendly piece is present on the tile
        **Parameters**
            # color: *str*, the color of the piece that is moving
        **Returns**
            # *bool* - True if a friendly piece is present, False otherwise
        '''
        return self.piece_present() and self.piece.color == color


    @staticmethod
    def on_board(*args):
        '''
        Checks to see if a position is on the board
        **Parameters**
            # *args: *int*, the position to check
        **Returns**
            # *bool* - True if the position is on the board, False otherwise
        '''
        for arg in args:
            if arg < 0 or arg > 7:
                return False
        return True


class Board:
    '''
    This class represents the board
    **Attributes**
        # tiles: a 2D list of tile objects
        # state: the state of the board that uses FEN notation to represent the board
    **Methods**
        # __init__: constructor for the Board class
        # create_board: creates the board object
        # create_pieces: creates the piece objects
        # move_piece: moves a piece on the board object between two tiles
    '''

    def __init__(self):
        '''
        This is the constructor for the Board class
        **Parameters**
            # none
        **Returns**
            # none
        '''

        self.tiles = [[0, 0, 0, 0, 0, 0, 0, 0] for col in range(COLS)]
        self.create_board()
        self.create_pieces('white')
        self.create_pieces('black')
        self.state = ""
        self.tensor = torch.zeros((6,8,8), dtype=torch.int)
        self.white_moves = []
        self.black_moves = []
    
    def move_piece(self, piece, move, testing=False):
        '''
        Moves a piece on the board object between two tiles
        **Parameters**
            # piece: *Piece obj*, the piece to move
            # move: *Move obj*, the move to make
            # testing: *bool*, True if the move is being made for testing, False otherwise
        **Returns**
            # none
        '''

        initial = move.init_tile
        final = move.final_tile
        
        en_passant_tile = self.tiles[final.row][final.col].empty_tile()

        # update the board, remove the initial piece
        self.tiles[initial.row][initial.col].piece = None
        # add the piece to the final tile
        self.tiles[final.row][final.col].piece = piece

        if isinstance(piece, Pawn):
            # method for en passant
            
            diff = final.col - initial.col
            if diff != 0 and en_passant_tile:
                self.tiles[initial.row][initial.col + diff].piece = None
                self.tiles[final.row][final.col].piece = piece

            # method for pawn promotion
            elif final.row == 0 or final.row == 7:
                self.tiles[final.row][final.col].piece = Queen(piece.color)

        # method for castling
        if isinstance(piece, King):
            if self.castle(initial, final) and not testing:
                # check to see what direction the king is castling
                diff = final.col - initial.col
                if diff < 0:
                    # castle left
                    rook = piece.l_rook
                else:
                    # castle right
                    rook = piece.r_rook
                self.move_piece(rook, rook.moves[-1])
                    
        # update that the piece has moved
        piece.moved = True

        # remove the valid moves
        piece.clear_moves()


    def castle(self, initial, final):
        '''
        Checks to see if the king is castling
        **Parameters**
            # initial: *Tile obj*, the initial tile of the king
            # final: *Tile obj*, the final tile of the king
        **Returns**
            # *bool* - True if the king is castling, False otherwise
        '''

        # check to see if the king is castling
        return abs(initial.col - final.col) == 2
    
    def legal_passant(self, piece):
        '''
        Checks to see if the en passant move is legal
        **Parameters**
            # piece: *Piece obj*, the piece to move
        **Returns**
            # *bool*: True if the en passant move is legal, False otherwise
            # en_passant: *bool*, changes the en passant attribute of the pawn
        '''
        
        if not isinstance(piece, Pawn):
            return
        
        for row in range(ROWS):
            for col in range(COLS):
                if isinstance(self.tiles[row][col].piece, Pawn):
                    self.tiles[row][col].piece.en_passant = False

        piece.en_passant = True


    def king_check(self, piece, move):
        '''
        Checks to see if the king is in check
        **Parameters**
            # piece: *Piece obj*, the piece to move
            # move: *Move obj*, the move to make
        **Returns**
            # *bool*: True if the king is in check, False otherwise
        '''

        # create a board copy
        board_copy = copy.deepcopy(self)
        piece_copy = copy.deepcopy(piece)
        # move the piece on the board copy
        board_copy.move_piece(piece_copy, move, testing=True)
        # check to see if the king is in check
        for i in range(ROWS):
            for j in range(COLS):
                # check to see if the tile has an enemy piece
                if board_copy.tiles[i][j].enemy_present(piece.color):
                    atk_piece = board_copy.tiles[i][j].piece
                    board_copy.valid_moves(atk_piece, i, j, k_check=False)
                    # loop the moves of the enemy piece to see if the king is in check
                    for atk_move in atk_piece.moves:
                        if isinstance(atk_move.final_tile.piece, King):
                            return True # if the king is in check
                        
        return False # if the king is not in check


    def check_valid(self, piece, move):
        '''
        Checks to see if the move is valid
        **Parameters**
            # piece: *Piece obj*, the piece to move
            # move: *Move obj*, the move to make
        **Returns**
            # *bool*: True if the move is in the list of moves for a piece, False otherwise
        '''
        return move in piece.moves


    def valid_moves(self, piece, row, col, k_check=True):
        '''
        Generates the valid moves for a piece
        **Parameters**
            # piece: *Piece obj*, the piece to move
            # row: *int*, the row of the piece
            # col: *int*, the column of the piece
            # k_check: *bool*, True if the king is in check, False otherwise
        **Returns**
            # piece.moves: *lst*, the list of valid moves for a piece
        '''

        def iterate_moves(directions):
            '''
            Iterates through the directions of a piece for pieces that move in a straight line
            **Parameters**
                # directions: *lst*, the list of directions for a piece
            **Returns**
                # none
            '''
            for direction in directions:
                    row_dir, col_dir = direction
                    pos_move_row = row + row_dir
                    pos_move_col = col + col_dir

                    while True:
                        if Tile.on_board(pos_move_row, pos_move_col):
                            
                            # get the initial and chosen tiles
                            initial = Tile(row, col)
                            # create a final piece to see if king is in check
                            final_piece = self.tiles[pos_move_row][pos_move_col].piece
                            chosen = Tile(pos_move_row, pos_move_col, final_piece)
                            move = Move(initial, chosen)

                            # check to see if the tile is empty
                            # continue iterating for blank tiles
                            # append the move if the king is not in check
                            if self.tiles[pos_move_row][pos_move_col].empty_tile():
                                if k_check:
                                    if not self.king_check(piece, move):
                                        piece.add_move(move)
                                    else:
                                        break
                                else:
                                    piece.add_move(move)
                            
                            # check to see if the tile has an enemy piece
                            # append move if the king is not in check
                            # break after adding move
                            elif self.tiles[pos_move_row][pos_move_col].enemy_present(piece.color):
                                if k_check:
                                    if not self.king_check(piece, move):
                                        piece.add_move(move)
                                    else:
                                        break
                                else:
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
        # dir is pos for black pawn, neg for white pawn

            # move pawn forward by 1 if moved
            # move pawn forward by 2 if not moved
            if piece.moved:
                vert_move = 1
            else:
                vert_move = 2

            # add moves for forward movement
            start_row = row + piece.dir
            end_row = row + (piece.dir * (1 + vert_move))
            for poss_row in range(start_row, end_row, piece.dir):
                if Tile.on_board(poss_row):
                    if self.tiles[poss_row][col].empty_tile():
                        init_tile = Tile(row, col)
                        final_tile = Tile(poss_row, col)
                        move = Move(init_tile, final_tile)

                        # check to see if the king is in check
                        if k_check:
                            if not self.king_check(piece, move):
                                piece.add_move(move)
                        else:
                            piece.add_move(move)

                    else:
                        break
                else:
                    break
            
            # add moves for diagonal movement
            poss_row = row + piece.dir
            poss_col = [col - 1, col + 1]
            for i in poss_col:
                if Tile.on_board(poss_row, i):
                    if self.tiles[poss_row][i].enemy_present(piece.color):
                        init_tile = Tile(row, col)
                        final_tile = Tile(poss_row, i)
                        # create a final piece to see if king is in check
                        final_piece = self.tiles[poss_row][i].piece

                        move = Move(init_tile, final_tile)

                        # check to see if the king is in check
                        if k_check:
                            if not self.king_check(piece, move):
                                piece.add_move(move)
                        else:
                            piece.add_move(move)

                    
            # code for en passant
            if piece.color == 'white':
                r_pawn = 3
                final_row = 2
            else:
                r_pawn = 4
                final_row = 5

            # left en passant
            if Tile.on_board(col - 1) and row == r_pawn:
                if self.tiles[row][col - 1].enemy_present(piece.color):
                    f_pawn = self.tiles[row][col - 1].piece
                    if isinstance(f_pawn, Pawn):
                        if f_pawn.en_passant:
                            init_tile = Tile(row, col)
                            final_tile = Tile(final_row, col - 1, f_pawn)
                            move = Move(init_tile, final_tile)
                        
                            # check to see if the king is in check
                            if k_check:
                                if not self.king_check(piece, move):
                                    piece.add_move(move)
                            else:
                                piece.add_move(move)

            # right en passant
            if Tile.on_board(col + 1) and row == r_pawn:
                if self.tiles[row][col + 1].enemy_present(piece.color):
                    f_pawn = self.tiles[row][col + 1].piece
                    if isinstance(f_pawn, Pawn):
                        if f_pawn.en_passant:
                            init_tile = Tile(row, col)
                            final_tile = Tile(final_row, col + 1, f_pawn)
                            move = Move(init_tile, final_tile)
                            
                            # check to see if the king is in check
                            if k_check:
                                if not self.king_check(piece, move):
                                    piece.add_move(move)
                            else:
                                piece.add_move(move)
                    

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
                        # add in a final piece to see if king in check
                        final_piece = self.tiles[pos_move_row][pos_move_col].piece
                        chosen = Tile(pos_move_row, pos_move_col, final_piece)
                        # make a move object
                        move = Move(initial, chosen)

                        # check if the move puts the king in check
                        if k_check:
                            if not self.king_check(piece, move):
                                piece.add_move(move)
                        else:
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
                        # append the move if it doesn't put the king in check
                        if k_check:
                            if not self.king_check(piece, move):
                                piece.add_move(move)
                        else:
                            piece.add_move(move)

            # castling
            # check to see if the king has moved
            if not piece.moved:
                
                # queen-side castle
                # check to see if the left rook has moved
                l_rook = self.tiles[row][0].piece
                if isinstance(l_rook, Rook):
                    if not l_rook.moved:
                        # iterate between rook and king to see if there are any pieces
                        for i in range(1, 4):
                            # if there are pieces, break
                            if self.tiles[row][i].piece_present():
                                break

                            # if there are no pieces, add the move
                            if i == 3:
                                # add the l_rook to the king's attributes
                                piece.l_rook = l_rook
                                # move rook
                                initial = Tile(row, 0)
                                chosen = Tile(row, 3)
                                move_rook = Move(initial, chosen)
                                # move king
                                initial = Tile(row, col)
                                chosen = Tile(row, 2)
                                move_king = Move(initial, chosen)

                                # check to see if the king is in check
                                # need to ensure that the rook does not cause check
                                # need to ensure that the king does not cause check
                                if k_check:
                                    if not self.king_check(piece, move_king) and not self.king_check(l_rook, move_rook):
                                        piece.add_move(move_king)
                                        l_rook.add_move(move_rook)
                                    else:
                                        break
                                else:
                                    piece.add_move(move_king)
                                    l_rook.add_move(move_rook)

                # king-side castle
                # check to see if the right rook has moved
                r_rook = self.tiles[row][7].piece
                if isinstance(r_rook, Rook):
                    if not r_rook.moved:
                        for i in range(5, 7):
                            if not self.tiles[row][i].empty_tile():
                                break
                            if i == 6:
                                # add the r_rook to the king's atrributes
                                piece.r_rook = r_rook
                                # move rook
                                initial = Tile(row, 7)
                                chosen = Tile(row, 5)
                                move_rook = Move(initial, chosen)

                                # move king
                                initial = Tile(row, col)
                                chosen = Tile(row, 6)
                                move_king = Move(initial, chosen)

                                # check to see if the king is in check
                                # need to ensure that the rook does not cause check
                                # need to ensure that the king does not cause check
                                if k_check:
                                    if not self.king_check(piece, move_king) and not self.king_check(r_rook, move_rook):
                                        piece.add_move(move_king)
                                        r_rook.add_move(move_rook)
                                    else:
                                        break
                                else:
                                    piece.add_move(move_king)
                                    r_rook.add_move(move_rook)


            # need to add check and checkmate
            # need to add stalemate

    def all_valid_moves(self, color):
        self.white_moves = []
        self.black_moves = []
        
        for row, _ in enumerate(self.tiles):
            for col, tile in enumerate(self.tiles[row]):
                if tile.piece_present() and tile.piece.color == color:
                    piece = tile.piece
                    self.valid_moves(piece, row, col)
                    for move in piece.moves:
                        # move_coordinates = ((move.init_tile.row, move.init_tile.col), 
                        #                     (move.final_tile.row, move.final_tile.col))
                        if color == 'white':
                            self.white_moves.append(move)
                        elif color == 'black':
                            self.black_moves.append(move)
                        piece.clear_moves()

        # while len(self.white_moves) != 4096 and len(self.black_moves) != 4096:
        #     if len(self.white_moves) != 0:
        #         self.white_moves.append(((0,0),(0,0)))
        #     elif len(self.black_moves) != 0:
        #         self.black_moves.append(((0,0),(0,0)))
        

    def create_board(self):
        '''
        Creates the board with tiles
        **Parameters**
            None
        **Returns**
            tiles: *list, list, Tile*, a 2D list of Tile objects
        '''

        for row in range(ROWS):
            for col in range(COLS):
                self.tiles[row][col] = Tile(row, col)


    def create_pieces(self, color):
        '''
        Creates the pieces for the board
        **Parameters**
            color: *str*, the color of the pieces
        **Returns**
            None
        '''

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
        # self.tiles[5][5] = Tile(5, 5, Pawn('black'))
        # self.tiles[6][6] = Tile(6, 6, Queen('black'))


class Drag:
    '''
    Class to handle the dragging of pieces
    **Attributes**
        mouse_x: *int*, the x coordinate of the mouse
        mouse_y: *int*, the y coordinate of the mouse
        init_col: *int*, the initial column of the piece
        init_row: *int*, the initial row of the piece
        piece: *Piece*, the piece that is being dragged
        dragging: *bool*, whether or not the piece is being dragged
    **Methods**
        __init__: constructor for the Drag class
        update_blit: updates the image when its being dragged
        update_mouse: updates the mouse position
    '''

    def __init__(self):
        '''
        Constructor for the Drag class
        **Parameters**
            None
        **Returns**
            None
        '''

        self.mouse_x = 0
        self.mouse_y = 0
        self.init_col = 0
        self.init_row = 0
        self.piece = None
        self.dragging = False


    def update_blit(self, surface):
        '''
        Updates the image when its being dragged
        **Parameters**
            surface: *pygame.Surface*, the surface to blit the image on
        **Returns**
            None
        '''

        # load the correct image
        img_path = self.piece.img
        img = pygame.image.load(img_path)
        # set the center of the image to the mouse position
        img_center = self.mouse_x, self.mouse_y
        self.piece.img_rect = img.get_rect(center=img_center)
        # blit the image
        surface.blit(img, self.piece.img_rect)


    def update_mouse(self, pos):
        '''
        Updates the mouse position when the mouse is moved
        **Parameters**
            pos: *tuple, int, int*, the position of the mouse (x, y)
        **Returns**
            None
        '''

        # position is a tuple representing coordinates of the mouse
        self.mouse_x = pos[0]
        self.mouse_y = pos[1]


    def initial_pos(self, pos):
        '''
        Initializes the initial position of the piece
        **Parameters**
            pos: *tuple, int, int*, the position of the mouse (x, y)
        **Returns**
            None
        '''

        self.init_col = pos[0] // TILE
        self.init_row = pos[1] // TILE


    def drag_piece(self, piece):
        '''
        Function to determine if a piece is being dragged
        **Parameters**
            piece: *Piece obj*, the piece object that is being dragged
        **Returns**
            none
        '''

        self.piece = piece
        self.dragging = True


    def drop_piece(self):
        '''
        Function to drop the piece
        **Parameters**
            None
        **Returns**
            None
        '''

        self.piece = None
        self.dragging = False


class Move:
    '''
    Class to handle moving pieces between tiles
    **Attributes**
        init_tile: *Tile*, the initial tile of the piece
        final_tile: *Tile*, the final tile of the piece
    **Methods**
        __init__: constructor for the Move class
        __eq__: determines if two moves are equal
    '''

    def __init__(self, init_tile, final_tile):
        '''
        Constructor for the Move class
        **Parameters**
            init_tile: *Tile*, the initial Tile object of the piece
            final_tile: *Tile*, the final Tile object of the piece
        **Returns**
            None
        '''

        self.init_tile = init_tile 
        self.final_tile = final_tile


    def __eq__(self, other):
        '''
        Equality function for the Move class
        **Parameters**
            other: *Move*, the other Move object to compare to
        **Returns**
            *bool*, whether or not the Move objects are equal
        '''

        return self.init_tile == other.init_tile and self.final_tile == other.final_tile

class ChessAI(nn.Module):
    def __init__(self, model_file=None, num_actions=4096):
        super(ChessAI, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=6, out_channels=32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1)
        self.fc1 = nn.Linear(in_features=128 * 8 * 8, out_features=256)
        self.fc2 = nn.Linear(in_features=256, out_features=num_actions)

        self.file = model_file

    def forward(self, state_tensor, num_moves):
        state_tensor = state_tensor.unsqueeze(0)
        state_tensor = F.relu(self.conv1(state_tensor))
        state_tensor = F.relu(self.conv2(state_tensor))
        state_tensor = F.relu(self.conv3(state_tensor))
        state_tensor = state_tensor.view(-1, 128 * 8 * 8)
        state_tensor = F.relu(self.fc1(state_tensor))
        state_tensor = self.fc2(state_tensor)

        # Resize output tensor to match the number of possible moves
        state_tensor = state_tensor.view(-1, num_moves)

        return F.softmax(state_tensor, dim=1)
    
    def train_model():
        # reset
            # Khalil made a reset function
        
        # reward
            # use algorithm to calculate score of position

        # game over function
            # checkmate or stalemate
        
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
    # chess.run()
    chess.train()

    # import chess
    # import chess.engine

    # # Change this if stockfish is somewhere else
    # engine = chess.engine.SimpleEngine.popen_uci("C:/Users/cgaul/Desktop/CBID 2022-2023/Software Carpentry/Chess/Chess/stockfish_15.1_win_x64_avx2/stockfish-windows-2022-x86-64-avx2.exe")

    # # The position represented in FEN
    # # board = chess.Board("rnbqkbnr/pp1ppppp/8/2p5/4P3/5N2/PPPP1PPP/RNBQKB1R b KQkq - 1 2")
    # board = chess.Board("r1b1k1nr/pppp1ppp/2n5/8/8/8/PPPP1PPP/RNBQKBNR w KQkq - 0 1")
    

    # # Limit our search so it doesn't run forever
    # info = engine.analyse(board, chess.engine.Limit(depth=20))

    # # Get the centipawn score from the evaluation
    # score = info["score"].relative.score(mate_score=100000) / 100

    # print("getting info")
    # print(f"Centipawn score: {score}")
    # print("info got")  

    # engine.close()
