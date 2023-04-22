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
        self.playable_grid = pd.DataFrame([[(0,0) for x in range(8)] for y in range(8)],
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
        self.starting_position = starting_position
        # Initialize starting position
        self.position = starting_position
        # Initialize type of the piece
        self.type_of_piece = type_of_piece

         # Set up inital position of piece on the board
        self.place_piece(starting_position)

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
        # Initialize the current position of the piece
        current_file = self.get_position()[0]
        current_rank = self.get_position()[1]

        # Reset the square that the piece is leaving
        board.playable_grid[current_file][current_rank] = (0,0)

        # Place a piece on the board depending on the type
        # Place a white piece
        if self.color == "white":
            # If a king place a 1
            if self.type_of_piece == "king":
                board.playable_grid[coordinate[0]][coordinate[1]] = ("white","king")
            # If a queen place a 2
            if self.type_of_piece == "queen":
                board.playable_grid[coordinate[0]][coordinate[1]] = ("white","queen")
            # If a rook place a 3
            if self.type_of_piece == "rook":
                board.playable_grid[coordinate[0]][coordinate[1]] = ("white","rook")
            # If a bishop place a 4
            if self.type_of_piece == "bishop":
                board.playable_grid[coordinate[0]][coordinate[1]] = ("white","bishop")
            # If a knight place a 5
            if self.type_of_piece == "knight":
                board.playable_grid[coordinate[0]][coordinate[1]] = ("white","knight")
            # If a pawn place a 6
            if self.type_of_piece == "pawn":
                board.playable_grid[coordinate[0]][coordinate[1]] = ("white","pawn")
        # Place a black piece
        if self.color == "black":
            # If a king place a 1
            if self.type_of_piece == "king":
                board.playable_grid[coordinate[0]][coordinate[1]] = ("black","king")
            # If a queen place a 2
            if self.type_of_piece == "queen":
                board.playable_grid[coordinate[0]][coordinate[1]] = ("black","queen")
            # If a rook place a 3
            if self.type_of_piece == "rook":
                board.playable_grid[coordinate[0]][coordinate[1]] = ("black","rook")
            # If a bishop place a 4
            if self.type_of_piece == "bishop":
                board.playable_grid[coordinate[0]][coordinate[1]] = ("black","bishop")
            # If a knight place a 5
            if self.type_of_piece == "knight":
                board.playable_grid[coordinate[0]][coordinate[1]] = ("black","knight")
            # If a pawn place a 6
            if self.type_of_piece == "pawn":
                board.playable_grid[coordinate[0]][coordinate[1]] = ("black","pawn")

        # Update the position that the piece is placed
        self.set_position(coordinate)

    def legal_moves(self):
        '''
        This will calculate and return the availible squares that the
        piece could legally move to

        **Parameters**
            None

        **Returns**
            legal_moves: *list, tuples*
                A list of all the squares a piece could move to
        '''
        # Initialize the legal_moves list
        legal_moves = []

        legal_files = ["A","B","C","D","E","F","G","H"]
        legal_ranks = [1,2,3,4,5,6,7,8]

        # If the type of piece is a pawn
        if self.type_of_piece == "pawn":
            # If the color of the pawn is white
            if self.color == "white":
                # Iniitalize the inital position of the file
                position_file = self.get_position()[0]
                # Initialize the rank in front of the pawn
                position_rank_ahead = self.get_position()[1] + 1

                # For the positions in front of the pawn keep going while the sqaure
                # is empty
                while position_file in legal_files\
                    and position_rank_ahead in legal_ranks\
                    and board.playable_grid[position_file][position_rank_ahead] == (0,0):
                    # Append this move as a legal move
                    legal_moves.append((position_file,position_rank_ahead))
                    # If the pawn is in the starting position
                    if self.get_position() == self.starting_position:
                        # Allow for two moves to be added to the legal_moves list
                        if len(legal_moves) == 2:
                            break
                    # If the pawn is in any other position
                    else:
                        # Allow for one move to be added to the legal_moves list
                        if len(legal_moves) == 1:
                            break
                    # Check another square ahead
                    position_rank_ahead = position_rank_ahead + 1

                # Initialize positions to check if pawn is attacking a piece
                position_file_attack_1 = chr(ord(position_file)+1)
                position_file_attack_2 = chr(ord(position_file)-1)
                position_rank_attack = self.get_position()[1] + 1

                # If there is a black piece in either attack position add that position
                # to the legal_moves list
                if  position_file_attack_1 in legal_files\
                    and position_rank_attack in legal_ranks\
                    and board.playable_grid[position_file_attack_1]\
                                           [position_rank_attack][0] == "black":
                    legal_moves.append((position_file_attack_1,position_rank_attack))
                if  position_file_attack_2 in legal_files\
                    and position_rank_attack in legal_ranks\
                    and board.playable_grid[position_file_attack_2]\
                                           [position_rank_attack][0] == "black":
                    legal_moves.append((position_file_attack_2,position_rank_attack))

            # If the color of the pawn is black
            if self.color == "black":
                # Iniitalize the inital position of the file
                position_file = self.get_position()[0]
                # Initialize the rank in front of the pawn
                position_rank_behind = self.get_position()[1] - 1

                # For the positions in front of the pawn keep going while the sqaure
                # is empty
                while position_file in legal_files\
                    and position_rank_behind in legal_ranks\
                    and board.playable_grid[position_file][position_rank_behind] == (0,0):
                    # Append this move as a legal move
                    legal_moves.append((position_file,position_rank_behind))
                    # If the pawn is in the starting position
                    if self.get_position() == self.starting_position:
                        # Allow for two moves to be added to the legal_moves list
                        if len(legal_moves) == 2:
                            break
                    # If the pawn is in any other position
                    else:
                        # Allow for one move to be added to the legal_moves list
                        if len(legal_moves) == 1:
                            break
                    # Check another square ahead
                    position_rank_ahead = position_rank_behind - 1

                # Initialize positions to check if pawn is attacking a piece
                position_file_attack_1 = chr(ord(position_file)+1)
                position_file_attack_2 = chr(ord(position_file)-1)
                position_rank_attack = self.get_position()[1] - 1

                # If there is a black piece in either attack position add that position
                # to the legal_moves list
                if position_file_attack_1 in legal_files\
                    and position_rank_attack in legal_ranks\
                    and board.playable_grid[position_file_attack_1]\
                                           [position_rank_attack][0] == "white":
                    legal_moves.append((position_file_attack_1,position_rank_attack))
                if position_file_attack_2 in legal_files\
                    and position_rank_attack in legal_ranks\
                    and board.playable_grid[position_file_attack_2]\
                                           [position_rank_attack][0] == "white":
                    legal_moves.append((position_file_attack_2,position_rank_attack))

        # If the type of piece is a bishop
        if self.type_of_piece == "bishop":
            # If the color of the bishop is white
            if self.color == 'white':
                ## Direction increase rank, increase file
                # Iniitalize the file ahead of the bishop
                position_file_ahead = chr(ord(self.get_position()[0]) + 1)
                # Initialize the rank in front of the bishop
                position_rank_ahead = self.get_position()[1] + 1
                # Initialize piece seen variable to 0
                piece_seen = 0

                # For the positions in front of the bishop keep going while the sqaure
                # is empty
                while position_file_ahead in legal_files\
                    and position_rank_ahead in legal_ranks\
                    and board.playable_grid[position_file_ahead]\
                                           [position_rank_ahead][0] != "white"\
                    and piece_seen == 0:
                    # Check if a piece of the opposite color was seen
                    if board.playable_grid[position_file_ahead][position_rank_ahead][0] == "black":
                        piece_seen += 1
                    # Append this move as a legal move
                    legal_moves.append((position_file_ahead,position_rank_ahead))
                    # Check another square ahead
                    position_file_ahead = chr(ord(position_file_ahead) + 1)
                    position_rank_ahead = position_rank_ahead + 1

                ## Direction decrease rank, increase file
                # Iniitalize the file ahead of the bishop
                position_file_ahead = chr(ord(self.get_position()[0]) + 1)
                # Initialize the rank in behind of the bishop
                position_rank_behind = self.get_position()[1] - 1
                # Initialize piece seen variable to 0
                piece_seen = 0

                # For the positions in front of the bishop keep going while the sqaure
                # is empty
                while position_file_ahead in legal_files\
                    and position_rank_behind in legal_ranks\
                    and board.playable_grid[position_file_ahead]\
                                           [position_rank_behind][0] != "white"\
                    and piece_seen == 0:
                    # Check if a piece of the opposite color was seen
                    if board.playable_grid[position_file_ahead][position_rank_behind][0] == "black":
                        piece_seen += 1
                    # Append this move as a legal move
                    legal_moves.append((position_file_ahead,position_rank_behind))
                    # Check another square ahead
                    position_file_ahead = chr(ord(position_file_ahead) + 1)
                    position_rank_behind = position_rank_behind - 1

                ## Direction increase rank, decrease file
                # Iniitalize the file behind the bishop
                position_file_behind = chr(ord(self.get_position()[0]) - 1)
                # Initialize the rank ahead of the bishop
                position_rank_ahead = self.get_position()[1] + 1
                # Initialize piece seen variable to 0
                piece_seen = 0

                # For the positions in front of the bishop keep going while the sqaure
                # is empty
                while position_file_behind in legal_files\
                    and position_rank_ahead in legal_ranks\
                    and board.playable_grid[position_file_behind]\
                                           [position_rank_ahead][0] != "white"\
                    and piece_seen == 0:
                    # Check if a piece of the opposite color was seen
                    if board.playable_grid[position_file_behind][position_rank_ahead][0] == "black":
                        piece_seen += 1
                    # Append this move as a legal move
                    legal_moves.append((position_file_behind,position_rank_ahead))
                    # Check another square ahead
                    position_file_behind = chr(ord(position_file_behind) - 1)
                    position_rank_ahead = position_rank_ahead + 1

                ## Direction decrease rank, decrease file
                # Iniitalize the inital position of the file
                position_file_behind = chr(ord(self.get_position()[0]) - 1)
                # Initialize the rank in front of the bishop
                position_rank_behind = self.get_position()[1] - 1
                # Initialize piece seen variable to 0
                piece_seen = 0

                # For the positions in front of the bishop keep going while the sqaure
                # is empty
                while position_file_behind in legal_files\
                    and position_rank_behind in legal_ranks\
                    and board.playable_grid[position_file_behind]\
                                           [position_rank_behind][0] != "white"\
                    and piece_seen == 0:
                    # Check if a piece of the opposite color was seen
                    if board.playable_grid[position_file_behind]\
                                          [position_rank_behind][0] == "black":
                        piece_seen += 1
                    # Append this move as a legal move
                    legal_moves.append((position_file_behind,position_rank_behind))
                    # Check another square ahead
                    position_file_behind = chr(ord(position_file_behind) - 1)
                    position_rank_behind = position_rank_behind - 1

            # If the color of the rook is black
            if self.color == 'black':
                ## Direction increase rank, increase file
                # Iniitalize the file ahead of the bishop
                position_file_ahead = chr(ord(self.get_position()[0]) + 1)
                # Initialize the rank in front of the bishop
                position_rank_ahead = self.get_position()[1] + 1
                # Initialize piece seen variable to 0
                piece_seen = 0

                # For the positions in front of the bishop keep going while the sqaure
                # is empty
                while position_file_ahead in legal_files\
                    and position_rank_ahead in legal_ranks\
                    and board.playable_grid[position_file_ahead][position_rank_ahead][0] != "black"\
                    and piece_seen == 0:
                    # Check if a piece of the opposite color was seen
                    if board.playable_grid[position_file_ahead][position_rank_ahead][0] == "white":
                        piece_seen += 1
                    # Append this move as a legal move
                    legal_moves.append((position_file_ahead,position_rank_ahead))
                    # Check another square ahead
                    position_file_ahead = chr(ord(position_file_ahead) + 1)
                    position_rank_ahead = position_rank_ahead + 1

                ## Direction decrease rank, increase file
                # Iniitalize the file ahead of the bishop
                position_file_ahead = chr(ord(self.get_position()[0]) + 1)
                # Initialize the rank in behind of the bishop
                position_rank_behind = self.get_position()[1] - 1
                # Initialize piece seen variable to 0
                piece_seen = 0

                # For the positions in front of the bishop keep going while the sqaure
                # is empty
                while position_file_ahead in legal_files\
                    and position_rank_behind in legal_ranks\
                    and board.playable_grid[position_file_ahead]\
                                           [position_rank_behind][0] != "black"\
                    and piece_seen == 0:
                    # Check if a piece of the opposite color was seen
                    if board.playable_grid[position_file_ahead][position_rank_behind][0] == "white":
                        piece_seen += 1
                    # Append this move as a legal move
                    legal_moves.append((position_file_ahead,position_rank_behind))
                    # Check another square ahead
                    position_file_ahead = chr(ord(position_file_ahead) + 1)
                    position_rank_behind = position_rank_behind - 1

                ## Direction increase rank, decrease file
                # Iniitalize the file behind the bishop
                position_file_behind = chr(ord(self.get_position()[0]) - 1)
                # Initialize the rank ahead of the bishop
                position_rank_ahead = self.get_position()[1] + 1
                # Initialize piece seen variable to 0
                piece_seen = 0

                # For the positions in front of the bishop keep going while the sqaure
                # is empty
                while position_file_behind in legal_files\
                    and position_rank_ahead in legal_ranks\
                    and board.playable_grid[position_file_behind]\
                                           [position_rank_ahead][0] != "black"\
                    and piece_seen == 0:
                    # Check if a piece of the opposite color was seen
                    if board.playable_grid[position_file_behind][position_rank_ahead][0] == "white":
                        piece_seen += 1
                    # Append this move as a legal move
                    legal_moves.append((position_file_behind,position_rank_ahead))
                    # Check another square ahead
                    position_file_behind = chr(ord(position_file_behind) - 1)
                    position_rank_ahead = position_rank_ahead + 1

                ## Direction decrease rank, decrease file
                # Iniitalize the inital position of the file
                position_file_behind = chr(ord(self.get_position()[0]) - 1)
                # Initialize the rank in front of the bishop
                position_rank_behind = self.get_position()[1] - 1
                # Initialize piece seen variable to 0
                piece_seen = 0

                # For the positions in front of the bishop keep going while the sqaure
                # is empty
                while position_file_behind in legal_files\
                    and position_rank_behind in legal_ranks\
                    and board.playable_grid[position_file_behind]\
                                           [position_rank_behind][0] != "black"\
                    and piece_seen == 0:
                    # Check if a piece of the opposite color was seen
                    if board.playable_grid[position_file_behind]\
                                          [position_rank_behind][0] == "white":
                        piece_seen += 1
                    # Append this move as a legal move
                    legal_moves.append((position_file_behind,position_rank_behind))
                    # Check another square ahead
                    position_file_behind = chr(ord(position_file_behind) - 1)
                    position_rank_behind = position_rank_behind - 1

        # If the type of piece is a rook
        if self.type_of_piece == "rook":
            # If the color of the rook is white
            if self.color == 'white':
                ## Direction increase rank
                # Iniitalize the inital position of the file
                position_file = self.get_position()[0]
                # Initialize the rank in front of the pawn
                position_rank_ahead = self.get_position()[1] + 1
                # Initialize piece seen variable to 0
                piece_seen = 0

                # For the positions in front of the rook keep going while the sqaure
                # is empty
                while position_file in legal_files\
                    and position_rank_ahead in legal_ranks\
                    and board.playable_grid[position_file]\
                                           [position_rank_ahead][0] != "white"\
                    and piece_seen == 0:
                    # Check if a piece of the opposite color was seen
                    if board.playable_grid[position_file][position_rank_ahead][0] == "black":
                        piece_seen += 1
                    # Append this move as a legal move
                    legal_moves.append((position_file,position_rank_ahead))
                    # Check another square ahead
                    position_rank_ahead = position_rank_ahead + 1

                ## Direction decrease rank
                # Iniitalize the inital position of the file
                position_file = self.get_position()[0]
                # Initialize the rank in front of the pawn
                position_rank_behind = self.get_position()[1] - 1
                # Initialize piece seen variable to 0
                piece_seen = 0

                # For the positions in front of the rook keep going while the sqaure
                # is empty
                while position_file in legal_files\
                    and position_rank_behind in legal_ranks\
                    and board.playable_grid[position_file][position_rank_behind][0] != "white"\
                    and piece_seen == 0:
                    # Check if a piece of the opposite color was seen
                    if board.playable_grid[position_file][position_rank_behind][0] == "black":
                        piece_seen += 1
                    # Append this move as a legal move
                    legal_moves.append((position_file,position_rank_behind))
                    # Check another square ahead
                    position_rank_behind = position_rank_behind - 1

                ## Direction increase file
                # Iniitalize the inital position of the file
                position_file_ahead = chr(ord(self.get_position()[0]) + 1)
                # Initialize the rank in front of the rook
                position_rank = self.get_position()[1]
                # Initialize piece seen variable to 0
                piece_seen = 0

                # For the positions in front of the rook keep going while the sqaure
                # is empty
                while position_file_ahead in legal_files\
                    and position_rank in legal_ranks\
                    and board.playable_grid[position_file_ahead][position_rank][0] != "white"\
                    and piece_seen == 0:
                    # Check if a piece of the opposite color was seen
                    if board.playable_grid[position_file_ahead][position_rank][0] == "black":
                        piece_seen += 1
                    # Append this move as a legal move
                    legal_moves.append((position_file_ahead,position_rank))
                    # Check another square ahead
                    position_file_ahead = chr(ord(position_file_ahead) + 1)

                ## Direction decrease file
                # Iniitalize the inital position of the file
                position_file_behind = chr(ord(self.get_position()[0]) - 1)
                # Initialize the rank in front of the rook
                position_rank = self.get_position()[1]
                # Initialize piece seen variable to 0
                piece_seen = 0

                # For the positions in front of the rook keep going while the sqaure
                # is empty
                while position_file_behind in legal_files\
                    and position_rank in legal_ranks\
                    and board.playable_grid[position_file_behind][position_rank][0] != "white"\
                    and piece_seen == 0:
                    # Check if a piece of the opposite color was seen
                    if board.playable_grid[position_file_behind][position_rank][0] == "black":
                        piece_seen += 1
                    # Append this move as a legal move
                    legal_moves.append((position_file_behind,position_rank))
                    # Check another square ahead
                    position_file_behind = chr(ord(position_file_behind) - 1)
            # If the color of the rook is black
            if self.color == 'black':
                ## Direction increase rank
                # Iniitalize the inital position of the file
                position_file = self.get_position()[0]
                # Initialize the rank in front of the rook
                position_rank_ahead = self.get_position()[1] + 1
                # Initialize piece seen variable to 0
                piece_seen = 0

                # For the positions in front of the rook keep going while the sqaure
                # is empty
                while position_file in legal_files\
                    and position_rank_ahead in legal_ranks\
                    and board.playable_grid[position_file][position_rank_ahead][0] != "black"\
                    and piece_seen == 0:
                    # Check if a piece of the opposite color was seen
                    if board.playable_grid[position_file][position_rank_ahead][0] == "white":
                        piece_seen += 1
                    # Append this move as a legal move
                    legal_moves.append((position_file,position_rank_ahead))
                    # Check another square ahead
                    position_rank_ahead = position_rank_ahead + 1

                ## Direction decrease rank
                # Iniitalize the inital position of the file
                position_file = self.get_position()[0]
                # Initialize the rank in front of the rook
                position_rank_behind = self.get_position()[1] - 1
                # Initialize piece seen variable to 0
                piece_seen = 0

                # For the positions in front of the rook keep going while the sqaure
                # is empty
                while position_file in legal_files\
                    and position_rank_behind in legal_ranks\
                    and board.playable_grid[position_file][position_rank_behind][0] != "black"\
                    and piece_seen == 0:
                    # Check if a piece of the opposite color was seen
                    if board.playable_grid[position_file][position_rank_behind][0] == "white":
                        piece_seen += 1
                    # Append this move as a legal move
                    legal_moves.append((position_file,position_rank_behind))
                    # Check another square ahead
                    position_rank_behind = position_rank_behind - 1

                ## Direction increase file
                # Iniitalize the inital position of the file
                position_file_ahead = chr(ord(self.get_position()[0]) + 1)
                # Initialize the rank in front of the rook
                position_rank = self.get_position()[1]
                # Initialize piece seen variable to 0
                piece_seen = 0

                # For the positions in front of the rook keep going while the sqaure
                # is empty
                while position_file_ahead in legal_files\
                    and position_rank in legal_ranks\
                    and board.playable_grid[position_file_ahead][position_rank][0] != "black"\
                    and piece_seen == 0:
                    # Check if a piece of the opposite color was seen
                    if board.playable_grid[position_file_ahead][position_rank][0] == "white":
                        piece_seen += 1
                    # Append this move as a legal move
                    legal_moves.append((position_file_ahead,position_rank))
                    # Check another square ahead
                    position_file_ahead = chr(ord(position_file_ahead) + 1)

                ## Direction decrease file
                # Iniitalize the inital position of the file
                position_file_behind = chr(ord(self.get_position()[0]) - 1)
                # Initialize the rank in front of the rook
                position_rank = self.get_position()[1]
                # Initialize piece seen variable to 0
                piece_seen = 0

                # For the positions in front of the rook keep going while the sqaure
                # is empty
                while position_file_behind in legal_files\
                    and position_rank in legal_ranks\
                    and board.playable_grid[position_file_behind][position_rank][0] != "black"\
                    and piece_seen == 0:
                    # Check if a piece of the opposite color was seen
                    if board.playable_grid[position_file_behind][position_rank][0] == "white":
                        piece_seen += 1
                    # Append this move as a legal move
                    legal_moves.append((position_file_behind,position_rank))
                    # Check another square ahead
                    position_file_behind = chr(ord(position_file_behind) - 1)

        # If the type of piece is a queen
        if self.type_of_piece == "queen":
            # If the color of the queen is white
            if self.color == 'white':
                ## Direction increase rank
                # Iniitalize the inital position of the file
                position_file = self.get_position()[0]
                # Initialize the rank in front of the queen
                position_rank_ahead = self.get_position()[1] + 1
                # Initialize piece seen variable to 0
                piece_seen = 0

                # For the positions in front of the queen keep going while the sqaure
                # is empty
                while position_file in legal_files\
                    and position_rank_ahead in legal_ranks\
                    and board.playable_grid[position_file]\
                                           [position_rank_ahead][0] != "white"\
                    and piece_seen == 0:
                    # Check if a piece of the opposite color was seen
                    if board.playable_grid[position_file][position_rank_ahead][0] == "black":
                        piece_seen += 1
                    # Append this move as a legal move
                    legal_moves.append((position_file,position_rank_ahead))
                    # Check another square ahead
                    position_rank_ahead = position_rank_ahead + 1

                ## Direction decrease rank
                # Iniitalize the inital position of the file
                position_file = self.get_position()[0]
                # Initialize the rank in front of the queen
                position_rank_behind = self.get_position()[1] - 1
                # Initialize piece seen variable to 0
                piece_seen = 0

                # For the positions in front of the queen keep going while the sqaure
                # is empty
                while position_file in legal_files\
                    and position_rank_behind in legal_ranks\
                    and board.playable_grid[position_file][position_rank_behind][0] != "white"\
                    and piece_seen == 0:
                    # Check if a piece of the opposite color was seen
                    if board.playable_grid[position_file][position_rank_behind][0] == "black":
                        piece_seen += 1
                    # Append this move as a legal move
                    legal_moves.append((position_file,position_rank_behind))
                    # Check another square behind
                    position_rank_behind = position_rank_behind - 1

                ## Direction increase file
                # Iniitalize the inital position of the file
                position_file_ahead = chr(ord(self.get_position()[0]) + 1)
                # Initialize the rank in front of the queen
                position_rank = self.get_position()[1]
                # Initialize piece seen variable to 0
                piece_seen = 0

                # For the positions in front of the queen keep going while the sqaure
                # is empty
                while position_file_ahead in legal_files\
                    and position_rank in legal_ranks\
                    and board.playable_grid[position_file_ahead][position_rank][0] != "white"\
                    and piece_seen == 0:
                    # Check if a piece of the opposite color was seen
                    if board.playable_grid[position_file_ahead][position_rank][0] == "black":
                        piece_seen += 1
                    # Append this move as a legal move
                    legal_moves.append((position_file_ahead,position_rank))
                    # Check another square ahead
                    position_file_ahead = chr(ord(position_file_ahead) + 1)

                ## Direction decrease file
                # Iniitalize the inital position of the file
                position_file_behind = chr(ord(self.get_position()[0]) - 1)
                # Initialize the rank in front of the queen
                position_rank = self.get_position()[1]
                # Initialize piece seen variable to 0
                piece_seen = 0

                # For the positions in front of the queen keep going while the sqaure
                # is empty
                while position_file_behind in legal_files\
                    and position_rank in legal_ranks\
                    and board.playable_grid[position_file_behind][position_rank][0] != "white"\
                    and piece_seen == 0:
                    # Check if a piece of the opposite color was seen
                    if board.playable_grid[position_file_behind][position_rank][0] == "black":
                        piece_seen += 1
                    # Append this move as a legal move
                    legal_moves.append((position_file_behind,position_rank))
                    # Check another square behind
                    position_file_behind = chr(ord(position_file_behind) - 1)

                ## Direction increase rank, increase file
                # Iniitalize the file ahead of the queen
                position_file_ahead = chr(ord(self.get_position()[0]) + 1)
                # Initialize the rank in front of the queen
                position_rank_ahead = self.get_position()[1] + 1
                # Initialize piece seen variable to 0
                piece_seen = 0

                # For the positions in front of the queen keep going while the sqaure
                # is empty
                while position_file_ahead in legal_files\
                    and position_rank_ahead in legal_ranks\
                    and board.playable_grid[position_file_ahead]\
                                           [position_rank_ahead][0] != "white"\
                    and piece_seen == 0:
                    # Check if a piece of the opposite color was seen
                    if board.playable_grid[position_file_ahead][position_rank_ahead][0] == "black":
                        piece_seen += 1
                    # Append this move as a legal move
                    legal_moves.append((position_file_ahead,position_rank_ahead))
                    # Check another square ahead
                    position_file_ahead = chr(ord(position_file_ahead) + 1)
                    position_rank_ahead = position_rank_ahead + 1

                ## Direction decrease rank, increase file
                # Iniitalize the file ahead of the queen
                position_file_ahead = chr(ord(self.get_position()[0]) + 1)
                # Initialize the rank in behind of the queen
                position_rank_behind = self.get_position()[1] - 1
                # Initialize piece seen variable to 0
                piece_seen = 0

                # For the positions in front of the queen keep going while the sqaure
                # is empty
                while position_file_ahead in legal_files\
                    and position_rank_behind in legal_ranks\
                    and board.playable_grid[position_file_ahead]\
                                           [position_rank_behind][0] != "white"\
                    and piece_seen == 0:
                    # Check if a piece of the opposite color was seen
                    if board.playable_grid[position_file_ahead][position_rank_behind][0] == "black":
                        piece_seen += 1
                    # Append this move as a legal move
                    legal_moves.append((position_file_ahead,position_rank_behind))
                    # Check another square ahead
                    position_file_ahead = chr(ord(position_file_ahead) + 1)
                    position_rank_behind = position_rank_behind - 1

                ## Direction increase rank, decrease file
                # Iniitalize the file behind the queen
                position_file_behind = chr(ord(self.get_position()[0]) - 1)
                # Initialize the rank ahead of the queen
                position_rank_ahead = self.get_position()[1] + 1
                # Initialize piece seen variable to 0
                piece_seen = 0

                # For the positions in front of the queen keep going while the sqaure
                # is empty
                while position_file_behind in legal_files\
                    and position_rank_ahead in legal_ranks\
                    and board.playable_grid[position_file_behind]\
                                           [position_rank_ahead][0] != "white"\
                    and piece_seen == 0:
                    # Check if a piece of the opposite color was seen
                    if board.playable_grid[position_file_behind][position_rank_ahead][0] == "black":
                        piece_seen += 1
                    # Append this move as a legal move
                    legal_moves.append((position_file_behind,position_rank_ahead))
                    # Check another square ahead
                    position_file_behind = chr(ord(position_file_behind) - 1)
                    position_rank_ahead = position_rank_ahead + 1

                ## Direction decrease rank, decrease file
                # Iniitalize the inital position of the file
                position_file_behind = chr(ord(self.get_position()[0]) - 1)
                # Initialize the rank in front of the queen
                position_rank_behind = self.get_position()[1] - 1
                # Initialize piece seen variable to 0
                piece_seen = 0

                # For the positions in front of the queen keep going while the sqaure
                # is empty
                while position_file_behind in legal_files\
                    and position_rank_behind in legal_ranks\
                    and board.playable_grid[position_file_behind]\
                                           [position_rank_behind][0] != "white"\
                    and piece_seen == 0:
                    # Check if a piece of the opposite color was seen
                    if board.playable_grid[position_file_behind]\
                                          [position_rank_behind][0] == "black":
                        piece_seen += 1
                    # Append this move as a legal move
                    legal_moves.append((position_file_behind,position_rank_behind))
                    # Check another square ahead
                    position_file_behind = chr(ord(position_file_behind) - 1)
                    position_rank_behind = position_rank_behind - 1
            # If the color of the queen is black
            if self.color == 'black':
                ## Direction increase rank
                # Iniitalize the inital position of the file
                position_file = self.get_position()[0]
                # Initialize the rank in front of the queen
                position_rank_ahead = self.get_position()[1] + 1
                # Initialize piece seen variable to 0
                piece_seen = 0

                # For the positions in front of the queen keep going while the sqaure
                # is empty
                while position_file in legal_files\
                    and position_rank_ahead in legal_ranks\
                    and board.playable_grid[position_file][position_rank_ahead][0] != "black"\
                    and piece_seen == 0:
                    # Check if a piece of the opposite color was seen
                    if board.playable_grid[position_file][position_rank_ahead][0] == "white":
                        piece_seen += 1
                    # Append this move as a legal move
                    legal_moves.append((position_file,position_rank_ahead))
                    # Check another square ahead
                    position_rank_ahead = position_rank_ahead + 1

                ## Direction decrease rank
                # Iniitalize the inital position of the file
                position_file = self.get_position()[0]
                # Initialize the rank in front of the queen
                position_rank_behind = self.get_position()[1] - 1
                # Initialize piece seen variable to 0
                piece_seen = 0

                # For the positions in front of the queen keep going while the sqaure
                # is empty
                while position_file in legal_files\
                    and position_rank_behind in legal_ranks\
                    and board.playable_grid[position_file][position_rank_behind][0] != "black"\
                    and piece_seen == 0:
                    # Check if a piece of the opposite color was seen
                    if board.playable_grid[position_file][position_rank_behind][0] == "white":
                        piece_seen += 1
                    # Append this move as a legal move
                    legal_moves.append((position_file,position_rank_behind))
                    # Check another square ahead
                    position_rank_behind = position_rank_behind - 1

                ## Direction increase file
                # Iniitalize the inital position of the file
                position_file_ahead = chr(ord(self.get_position()[0]) + 1)
                # Initialize the rank in front of the queen
                position_rank = self.get_position()[1]
                # Initialize piece seen variable to 0
                piece_seen = 0

                # For the positions in front of the queen keep going while the sqaure
                # is empty
                while position_file_ahead in legal_files\
                    and position_rank in legal_ranks\
                    and board.playable_grid[position_file_ahead][position_rank][0] != "black"\
                    and piece_seen == 0:
                    # Check if a piece of the opposite color was seen
                    if board.playable_grid[position_file_ahead][position_rank][0] == "white":
                        piece_seen += 1
                    # Append this move as a legal move
                    legal_moves.append((position_file_ahead,position_rank))
                    # Check another square ahead
                    position_file_ahead = chr(ord(position_file_ahead) + 1)

                ## Direction decrease file
                # Iniitalize the inital position of the file
                position_file_behind = chr(ord(self.get_position()[0]) - 1)
                # Initialize the rank in front of the queen
                position_rank = self.get_position()[1]
                # Initialize piece seen variable to 0
                piece_seen = 0

                # For the positions in front of the queen keep going while the sqaure
                # is empty
                while position_file_behind in legal_files\
                    and position_rank in legal_ranks\
                    and board.playable_grid[position_file_behind][position_rank][0] != "black"\
                    and piece_seen == 0:
                    # Check if a piece of the opposite color was seen
                    if board.playable_grid[position_file_behind][position_rank][0] == "white":
                        piece_seen += 1
                    # Append this move as a legal move
                    legal_moves.append((position_file_behind,position_rank))
                    # Check another square ahead
                    position_file_behind = chr(ord(position_file_behind) - 1)

                ## Direction increase rank, increase file
                # Iniitalize the file ahead of the queen
                position_file_ahead = chr(ord(self.get_position()[0]) + 1)
                # Initialize the rank in front of the queen
                position_rank_ahead = self.get_position()[1] + 1
                # Initialize piece seen variable to 0
                piece_seen = 0

                # For the positions in front of the queen keep going while the sqaure
                # is empty
                while position_file_ahead in legal_files\
                    and position_rank_ahead in legal_ranks\
                    and board.playable_grid[position_file_ahead][position_rank_ahead][0] != "black"\
                    and piece_seen == 0:
                    # Check if a piece of the opposite color was seen
                    if board.playable_grid[position_file_ahead][position_rank_ahead][0] == "white":
                        piece_seen += 1
                    # Append this move as a legal move
                    legal_moves.append((position_file_ahead,position_rank_ahead))
                    # Check another square ahead
                    position_file_ahead = chr(ord(position_file_ahead) + 1)
                    position_rank_ahead = position_rank_ahead + 1

                ## Direction decrease rank, increase file
                # Iniitalize the file ahead of the queen
                position_file_ahead = chr(ord(self.get_position()[0]) + 1)
                # Initialize the rank in behind of the queen
                position_rank_behind = self.get_position()[1] - 1
                # Initialize piece seen variable to 0
                piece_seen = 0

                # For the positions in front of the queen keep going while the sqaure
                # is empty
                while position_file_ahead in legal_files\
                    and position_rank_behind in legal_ranks\
                    and board.playable_grid[position_file_ahead]\
                                           [position_rank_behind][0] != "black"\
                    and piece_seen == 0:
                    # Check if a piece of the opposite color was seen
                    if board.playable_grid[position_file_ahead][position_rank_behind][0] == "white":
                        piece_seen += 1
                    # Append this move as a legal move
                    legal_moves.append((position_file_ahead,position_rank_behind))
                    # Check another square ahead
                    position_file_ahead = chr(ord(position_file_ahead) + 1)
                    position_rank_behind = position_rank_behind - 1

                ## Direction increase rank, decrease file
                # Iniitalize the file behind the queen
                position_file_behind = chr(ord(self.get_position()[0]) - 1)
                # Initialize the rank ahead of the queen
                position_rank_ahead = self.get_position()[1] + 1
                # Initialize piece seen variable to 0
                piece_seen = 0

                # For the positions in front of the queen keep going while the sqaure
                # is empty
                while position_file_behind in legal_files\
                    and position_rank_ahead in legal_ranks\
                    and board.playable_grid[position_file_behind]\
                                           [position_rank_ahead][0] != "black"\
                    and piece_seen == 0:
                    # Check if a piece of the opposite color was seen
                    if board.playable_grid[position_file_behind][position_rank_ahead][0] == "white":
                        piece_seen += 1
                    # Append this move as a legal move
                    legal_moves.append((position_file_behind,position_rank_ahead))
                    # Check another square ahead
                    position_file_behind = chr(ord(position_file_behind) - 1)
                    position_rank_ahead = position_rank_ahead + 1

                ## Direction decrease rank, decrease file
                # Iniitalize the inital position of the file
                position_file_behind = chr(ord(self.get_position()[0]) - 1)
                # Initialize the rank in front of the queen
                position_rank_behind = self.get_position()[1] - 1
                # Initialize piece seen variable to 0
                piece_seen = 0

                # For the positions in front of the queen keep going while the sqaure
                # is empty
                while position_file_behind in legal_files\
                    and position_rank_behind in legal_ranks\
                    and board.playable_grid[position_file_behind]\
                                           [position_rank_behind][0] != "black"\
                    and piece_seen == 0:
                    # Check if a piece of the opposite color was seen
                    if board.playable_grid[position_file_behind]\
                                          [position_rank_behind][0] == "white":
                        piece_seen += 1
                    # Append this move as a legal move
                    legal_moves.append((position_file_behind,position_rank_behind))
                    # Check another square ahead
                    position_file_behind = chr(ord(position_file_behind) - 1)
                    position_rank_behind = position_rank_behind - 1

        # Return the legal_moves list
        return legal_moves

if __name__ == '__main__':
    # Create the board
    board = Board()

    # Create the kings
    b_king = Piece("black", ("E", 8), "king")
    w_king = Piece("white", ("E", 1), "king")

    # Create the queens
    b_queen = Piece("black", ("D", 8), "queen")
    w_queen = Piece("white", ("D", 1), "queen")

    # Create the rooks
    b_rook_a = Piece("black", ("A", 8), "rook")
    b_rook_h = Piece("black", ("H", 8), "rook")
    w_rook_a = Piece("white", ("A", 1), "rook")
    w_rook_h = Piece("white", ("H", 1), "rook")

    # Create the bishops
    b_bishop_c = Piece("black", ("C", 8), "bishop")
    b_bishop_f = Piece("black", ("F", 8), "bishop")
    w_bishop_c = Piece("white", ("C", 1), "bishop")
    w_bishop_f = Piece("white", ("F", 1), "bishop")

    # Create the knights
    b_knight_b = Piece("black", ("B", 8), "knight")
    b_knight_g = Piece("black", ("G", 8), "knight")
    w_knight_b = Piece("white", ("B", 1), "knight")
    w_knight_g = Piece("white", ("G", 1), "knight")

    # Create the pawns
    b_pawn_a = Piece("black", ("A", 7), "pawn")
    b_pawn_b = Piece("black", ("B", 7), "pawn")
    b_pawn_c = Piece("black", ("C", 7), "pawn")
    b_pawn_d = Piece("black", ("D", 7), "pawn")
    b_pawn_e = Piece("black", ("E", 7), "pawn")
    b_pawn_f = Piece("black", ("F", 7), "pawn")
    b_pawn_g = Piece("black", ("G", 7), "pawn")
    b_pawn_h = Piece("black", ("H", 7), "pawn")

    w_pawn_a = Piece("white", ("A", 2), "pawn")
    w_pawn_b = Piece("white", ("B", 2), "pawn")
    w_pawn_c = Piece("white", ("C", 2), "pawn")
    w_pawn_d = Piece("white", ("D", 2), "pawn")
    w_pawn_e = Piece("white", ("E", 2), "pawn")
    w_pawn_f = Piece("white", ("F", 2), "pawn")
    w_pawn_g = Piece("white", ("G", 2), "pawn")
    w_pawn_h = Piece("white", ("H", 2), "pawn")

    print("\nColor of board")
    print(board.color_grid)
    print("\nPlayable board")
    print(board.playable_grid)
    print("\n")

    print("Legal Moves for w_pawn_a")
    print(w_pawn_a.legal_moves())

    b_pawn_b.place_piece(("B",3))
    b_queen.place_piece(("D",5))

    print("Legal Moves for w_pawn_a")
    print(w_pawn_a.legal_moves())

    print("Legal Moves for b_queen")
    print(b_queen.legal_moves())

    print("\nPlayable board")
    print(board.playable_grid)
    print("\n")
