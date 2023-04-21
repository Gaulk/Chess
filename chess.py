'''
Software Carpentry - Final Project
Khalil Merali and Carter Gaulke
This file works as a interactive chess game.
This game can be played by two humans, one human and a bot, or two bots
'''
## Imports
import pandas as pd

## Class for Board
class Board:
    '''
    This class is used to create the board object. This board will interact with the
    pieces to play the game

    **Attributes**
        None
    
    **Methods**
        __init__(self):
            Initializes the board object.
    '''
    def __init__(self):
        # Initialize variable to store file names
        column_names=["A","B","C","D","E","F","G","H"]

        # Create grid for the color of the sqaures of the board
        self.color_grid = pd.DataFrame([[0 for x in range(8)] for y in range(8)],
                                       columns=column_names)
        # Shift index to start at 1
        self.color_grid.index = self.color_grid.index + 1
        # Create alternating 1's and 0's in the color grid
        flip = 1
        for rank in range(1,9):
            for file in column_names:
                if flip < 0:
                    self.color_grid[file][rank] = 0
                if flip > 0:
                    self.color_grid[file][rank] = 1
                flip = flip*-1
            flip = flip*-1

        # Create grid for the playing of the game
        self.playable_grid = pd.DataFrame([[0 for x in range(8)] for y in range(8)],
                                          columns=column_names,
                                          dtype=object)
        # Shift index to start at 1
        self.playable_grid.index = self.playable_grid.index + 1

class Piece:
    '''
    This class is used to create the piece object. This pieces will interact with the
    board to play the game

    **Attributes**
        color:
            The color of the piece
        starting_position:
            The square the piece starts on
        type:
            The type of the piece
    
    **Methods**
        __init__(self):
            Initializes the board object.
    '''
    def __init__(self, color, starting_position, type_of_piece):
        # Initialize the attributes of the class
        # Initialize color
        self.color = color
        # Initialize starting position
        self.position = starting_position
        # Initialize type of the piece
        self.type_of_piece = type_of_piece

    def get_color(self):
        '''
        This will return the color of the piece

        **Parameters**
            None

        **Returns**
            self.color: *str*
                A string that is the color of the piece
        '''
        # Return the color of the piece
        return self.color

    def get_position(self):
        '''
        This will return the position of the piece

        **Parameters**
            None

        **Returns**
            self.position: *tuple*
                A tuple for the current location of the piece
        '''
        # Return the position of the piece
        return self.position

    def set_position(self, coordinate):
        '''
        This will set the position of the piece

        **Parameters**
            coordinate: *tuple*

        **Returns**
            None
        '''
        # Set the position equal to the given coordinate
        self.position = coordinate

    def get_type_of_piece(self):
        '''
        This will return the type of piece

        **Parameters**
            None

        **Returns**
            self.type_of_piece: *str*
                A string for the type of piece it is
        '''
        # Return the type the piece is
        return self.type_of_piece

    def place_piece(self, coordinate):
        '''
        This will place a piece on the board

        **Parameters**
            coordinate: *tuple*
                A tuple holding a square on the board this
                piece will be placed

        **Returns**
            None
        '''
        # Place a piece on the board depending on the type
        # Place a white piece
        if self.color == "white":
            # If a king place a 1
            if self.type_of_piece == "king":
                board.playable_grid[coordinate[0]][coordinate[1]] = (1,1)
            # If a queen place a 2
            if self.type_of_piece == "queen":
                board.playable_grid[coordinate[0]][coordinate[1]] = (1,2)
            # If a rook place a 3
            if self.type_of_piece == "rook":
                board.playable_grid[coordinate[0]][coordinate[1]] = (1,3)
            # If a bishop place a 4
            if self.type_of_piece == "bishop":
                board.playable_grid[coordinate[0]][coordinate[1]] = (1,4)
            # If a knight place a 5
            if self.type_of_piece == "knight":
                board.playable_grid[coordinate[0]][coordinate[1]] = (1,5)
            # If a pawn place a 6
            if self.type_of_piece == "pawn":
                board.playable_grid[coordinate[0]][coordinate[1]] = (1,6)
        # Place a black piece
        if self.color == "black":
            # If a king place a 1
            if self.type_of_piece == "king":
                board.playable_grid[coordinate[0]][coordinate[1]] = (0,1)
            # If a queen place a 2
            if self.type_of_piece == "queen":
                board.playable_grid[coordinate[0]][coordinate[1]] = (0,2)
            # If a rook place a 3
            if self.type_of_piece == "rook":
                board.playable_grid[coordinate[0]][coordinate[1]] = (0,3)
            # If a bishop place a 4
            if self.type_of_piece == "bishop":
                board.playable_grid[coordinate[0]][coordinate[1]] = (0,4)
            # If a knight place a 5
            if self.type_of_piece == "knight":
                board.playable_grid[coordinate[0]][coordinate[1]] = (0,5)
            # If a pawn place a 6
            if self.type_of_piece == "pawn":
                board.playable_grid[coordinate[0]][coordinate[1]] = (0,6)

        # Update the position that the piece is placed
        self.set_position(coordinate)
    
    def legal_moves(self):
        '''
        This will return the type of piece

        **Parameters**
            type_of_piece: *str*
                A string for the type of piece it is

        **Returns**
            legal_moves: *list, tuples*
                A list of all the squares a piece could move to
        '''
        # add code for this

if __name__ == '__main__':
    board = Board()

    print("\nColor of board")
    print(board.color_grid)
    print("\nPlayable board")
    print(board.playable_grid)
    print("\n")

    piece = Piece("white", ("A",1), "bishop")
    piece_position = piece.get_position()

    print(piece.get_type_of_piece())
    print(piece.get_color())
    print(piece.get_position())

    piece.place_piece(("B",1))

    print("\nPlayable board")
    print(board.playable_grid)

    print("\nPiece position")
    print(piece.get_position())
