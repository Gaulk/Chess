'''
NOTE: Please see the README file for instructions on how to run the program

This code contains the main game loop

It also contains several classes that are used to create the game and display the game in a pygame GUI
The following classes are included:
    # Game - represents the game
    # Board - represents the board
    # Tile - represents a tile on the board
    # Piece - represents a piece that can be placed on a tile, super class of all pieces types:
        # King - represents a king
        # Pawn - represents a pawn
        # Rook - represents a rook
        # Knight - represents a knight
        # Bishop - represents a bishop
        # Queen - represents a queen
    # Drag - represents the dragging of a piece
    # Chess_App - represents the main game loop
    # ChessAI - represents the AI
    # ChessAgent - represents the agent
    # Move - represents a move

    Updates to make:
        Priority of updates:
        1. Break code into different files
            a. Organize Github
                i. Add folders into the tree
            b. Game/GUI
            c. Setting up the board
                i. Piece architecture
                ii. Board itself
            d. Moving pieces
            e. Finding legal moves and calculating check
            f. Create FEN
                i. Use this to pass the position between files
                    a. Ehh maybe on this, other files would just have the functions
                        Could reference/call them
        2. Re-do how finding legal moves works
            a. make check function work on current game board.
                Then try a move and check if in "check"
            b. Don't need to re-calculate all legal moves each time.
                Only certain moves changes when 1 piece moves
                Create running list of moves essentially.
                i. When recalculating, recalculate these
                    a. Piece that moved
                    b. If that piece moved into a tile that was a legal move
                        i. Need to update other pieces, can only go up to this move
                        ii. Potentially to make simpler recalculate if affected
                        iii. Need to consider if a piece moves, it could open up
                            pieces behind it to have more legal moves, 
                            would need to calculate this
        3. Create easier to use GUI (app for iOS and android?)
            a. Make it look modern
                i. Use a library that can be used on android and iOS
                ii. Looks fire
            b. Will have the ability to control all things from GUI
                i. Which color to play
                ii. If playing bot or person
                iii. Create profiles? Log what your record/rating is?
                iv. Highlight the previous move
        4. Make into a Docker thing
            a. This will allow to be packaged and easily used by someone else... Khaleezy
        4. Update Bot, train better
            a. Need to create better scoring system
        5. Add functionality to export the current position
            a. This will be useful to when trying to interface with the in person board
            b. Add ability to intake a position from the board, and move a piece accordingly
        
'''

import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'

# import libraries
import pygame
import sys
import copy
import random

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

import chess
import chess.engine
import platform

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
        # train - trains the AI
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


    def run(self, type_of_game):
        '''
        This method runs the main game loop, displaying the board and pieces
        and allowing for user input.
        Also allows for the user to select the type of game to be played (PVP or Bot)
        **Parameters**
            # type_of_game - the type of game to be played (PVP or Bot)
        **Returns**
            # none
        '''

        screen = self.screen
        game = self.game
        drag = self.game.drag
        board = self.game.board
        self.game.type_of_game = type_of_game
        model = ChessAI()
        model.load_model()
        agent = ChessAgent(model)

        if self.game.type_of_game == 'PVP':
            while True:
                # show the board and pieces
                game.show_board(screen)
                game.highlight_moves(screen)
                game.show_pieces(screen)
                if game.endgame:
                    game.show_endgame(screen)

                if drag.dragging:
                    drag.update_blit(screen)

                # quit application if user clicks the X
                for event in pygame.event.get():

                    # allows for clicks
                    if event.type == pygame.MOUSEBUTTONDOWN and not game.endgame:
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
                    elif event.type == pygame.MOUSEMOTION and not game.endgame:
                        if drag.dragging:
                            drag.update_mouse(event.pos)
                            # stack the board, highlights and pieces
                            game.show_board(screen)
                            game.highlight_moves(screen)
                            game.show_pieces(screen)   
                            drag.update_blit(screen)

                    # allows for dropping a piece
                    elif event.type == pygame.MOUSEBUTTONUP and not game.endgame:
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

                                game.change_turn()

                                game.counter(self.game.count)
                                # game.get_state()

                                game.is_gameover()
                                if game.gameover != None:
                                    game.show_endgame(screen)
                                    game.endgame = True

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
                
                # update the display after every event
                pygame.display.update()

        if self.game.type_of_game == 'Bot':
            while True:
                # show the board and pieces
                game.show_board(screen)
                game.highlight_moves(screen)
                game.show_pieces(screen)
                if game.endgame:
                    game.show_endgame(screen)

                if drag.dragging:
                    drag.update_blit(screen)

                # sets events within the game
                for event in pygame.event.get():
                    # if 'r' is pressed, reset the game
                    if event.type == pygame.KEYDOWN and game.endgame:
                        if event.key == pygame.K_r:
                            game.reset_game()
                            game = self.game
                            board = self.game.board
                            drag = self.game.drag
                    
                    if event.type == pygame.QUIT:
                        pygame.quit()
                        sys.exit()

                    if game.player_color == 'white':
                        # allows for clicks
                        if event.type == pygame.MOUSEBUTTONDOWN and not game.endgame:
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
                        elif event.type == pygame.MOUSEMOTION and not game.endgame:
                            if drag.dragging:
                                drag.update_mouse(event.pos)
                                # stack the board, highlights and pieces
                                game.show_board(screen)
                                game.highlight_moves(screen)
                                game.show_pieces(screen)   
                                drag.update_blit(screen)

                        # allows for dropping a piece
                        elif event.type == pygame.MOUSEBUTTONUP and not game.endgame:
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

                                    game.change_turn()
                                    game.counter(self.game.count)

                                    # Get FEN
                                    game.get_state()
                                    state = game.board.state
                                    print(state)
                                    # Calculate reward
                                    # relative, always positive if good for that color
                                    game.get_centipawn(state)
                                    print(game.centipawn)

                                    game.is_gameover()
                                    if game.gameover != None:
                                        game.show_endgame(screen)
                                        game.endgame = True

                            drag.drop_piece()

                        # if 'r' is pressed, reset the game
                        elif event.type == pygame.KEYDOWN:
                            if event.key == pygame.K_r:
                                game.reset_game()
                                game = self.game
                                board = self.game.board
                                drag = self.game.drag
                        
                        # quit application if user clicks the X
                        elif event.type == pygame.QUIT:
                            pygame.quit()
                            sys.exit()

                    # # update the display after every event
                    # pygame.display.update()

                    if game.player_color == 'black' and not game.endgame:
                        board.all_valid_moves(game.player_color)
                        game.state_tensor()
                        current_state_tensor = game.board.tensor
                        current_len_moves = game.board.black_moves_len
                        action, state_value = agent.select_action(current_state_tensor, current_len_moves)
                        move = game.board.black_moves_bot[action]
                        piece = board.tiles[move.init_tile.row][move.init_tile.col].piece

                        # Move piece
                        board.move_piece(piece, move)

                        # Change whose turn it is
                        game.change_turn()
                        # Increase counter
                        game.counter(self.game.count)

                        # Show the board
                        game.show_board(screen)
                        game.show_pieces(screen)

                        # Calcualte if gameover
                        game.is_gameover()
                        if game.gameover != None:
                            game.show_endgame(screen)
                            game.endgame = True
                
                # update the display after every event
                pygame.display.update()

    def train(self):
        '''
        This function allows the user to train the model while displaying the board
        **Parameters**
            None
        **Returns**
            None
        '''
        screen = self.screen
        game = self.game
        board = self.game.board
        model = ChessAI()
        model.load_model()
        agent = ChessAgent(model)
        number_of_games = 1

        while True:
            # show the board and pieces
            game.show_board(screen)
            game.highlight_moves(screen)
            game.show_pieces(screen)

            # shows the endgame screen
            if game.endgame:
                    game.show_endgame(screen)
                    game.reset_game()
                    game = self.game
                    board = self.game.board
                    number_of_games += 1

            while game.endgame == False:
                # run the game while not gameover
                model.load_model()
                board.all_valid_moves(game.player_color)
                # Select move
                if game.player_color == 'white':
                    game.state_tensor()
                    current_state_tensor = game.board.tensor
                    current_len_moves = game.board.white_moves_len
                    action, state_value = agent.select_action(current_state_tensor, current_len_moves)
                    move = game.board.white_moves_bot[action]
                if game.player_color == 'black':
                    game.state_tensor()
                    current_state_tensor = game.board.tensor
                    current_len_moves = game.board.black_moves_len
                    action, state_value = agent.select_action(current_state_tensor, current_len_moves)
                    move = game.board.black_moves_bot[action]
                piece = board.tiles[move.init_tile.row][move.init_tile.col].piece

                # Move piece
                board.move_piece(piece, move)

                # Change whose turn it is
                game.change_turn()
                # Increase counter
                game.counter(self.game.count)

                # Show the board
                game.show_board(screen)
                game.show_pieces(screen)

                # Calcualte if gameover
                game.is_gameover()
                if game.gameover != None:
                    game.show_endgame(screen)
                    game.endgame = True

                # Get FEN
                game.get_state()
                state = game.board.state
                # print(state)
                # Calculate reward
                # relative, always positive if good for that color
                game.get_centipawn(state)
                # print(game.centipawn)
                
                reward = (-1 * game.centipawn) - (game.previous_centipawn)
                # print(f"reward: {reward}")
                game.state_tensor()
                next_state_tensor = game.board.tensor
                board.all_valid_moves(game.player_color)
                if game.player_color == 'white':
                    next_num_moves = game.board.black_moves_len
                if game.player_color == 'black':
                    next_num_moves = game.board.white_moves_len
                # Update the model
                agent.update_model(state_value, reward, next_state_tensor, next_num_moves, game.endgame)
                game.previous_centipawn = game.centipawn

                # save the model
                model.save_model()
                pygame.display.update()

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
        # gameover - the gameover screen
        # halfmoves - the number of halfmoves
        # fullmoves - the number of fullmoves
        # previous_centipawn - the previous centipawn
        # centipawn - the current centipawn score
        # endgame - the endgame screen
    **Methods**
        # __init__ - constructor for the game
        # show_board - shows the board
        # show_pieces - shows the pieces
        # highlight_moves - highlights the valid moves
        # change_turn - changes the turn
        # reset_game - resets the game
        # get_state - gets the state of the board
        # counter - counts the number of moves
        # get_centipawn - gets the centipawn score
        # state_tensor - gets the state tensor
        # is_gameover - checks if the game is over
        # show_endgame - shows the endgame screen
    '''

    def __init__(self):
        '''
        This method initializes the game
        **Parameters**
            # none
        **Returns**
            # none
        '''

        self.board = Board()
        self.drag = Drag()
        self.ChessAI = ChessAI()
        self.player_color = 'white'
        self.count = 0
        self.gameover = None
        self.halfmoves = 0
        self.fullmoves = 1
        self.previous_centipawn = 0
        self.centipawn = 0
        self.endgame = False
        self.type_of_game = ''


    def is_gameover(self):
        '''
        This method checks if the game is over
        Function will check if the game is over by checkmate, stalemate, or 50 move rule
        **Parameters**
            # none
        **Returns**
            # none
        '''
        if self.board.is_checkmate('black'):
            self.gameover = 'White wins'
        else:
            self.board.all_valid_moves('black')
            if self.board.black_moves == []:
                self.gameover = 'Draw'
        if self.board.is_checkmate('white'):
            self.gameover = 'Black wins'
        else:
            self.board.all_valid_moves('white')
            if self.board.white_moves == []:
                self.gameover = 'Draw'
        if self.halfmoves >= 50:
            self.gameover = 'Draw'


    def get_centipawn(self, FEN):
        '''
        This method gets the centipawn score from the stockfish engine
        This assists in rewarding the model
        **Parameters**
            # FEN - the FEN of the board
        **Returns**
            # none
        '''
        # Change this if stockfish is somewhere else
        engine = chess.engine.SimpleEngine.popen_uci("stockfish_15.1_win_x64_popcnt/stockfish-windows-2022-x86-64-modern.exe")

        # The position represented in FEN
        # board = chess.Board("rnbqkbnr/pp1ppppp/8/2p5/4P3/5N2/PPPP1PPP/RNBQKB1R b KQkq - 1 2")
        board = chess.Board(FEN)
        
        # Limit our search so it doesn't run forever
        info = engine.analyse(board, chess.engine.Limit(depth=14))

        # Get the centipawn score from the evaluation
        self.centipawn = info["score"].relative.score(mate_score=100000) / 100 
        # self.centipawn = info["score"].relative.score() / 100

        engine.close()


    def show_board(self, surface):
        '''
        This method shows the board and the row and column labels
        **Parameters**
            # surface: *pygame.Surface* - the surface to draw on
        **Returns**
            # none
        '''

        # draw the board
        for row in range(ROWS):
            for col in range(COLS):
                # set color of tile
                # if row + col is even, color is white
                # if row + col is odd, color is black
                if (row + col) % 2 == 0:
                    color = (247, 207, 164) # light brown
                else:
                    color = (199, 141, 83) # dark brown
             
                # draw square for each tile on the board
                rect = (col * TILE, row * TILE, TILE, TILE)
                pygame.draw.rect(surface, color, rect)
                

                ## Creating a label
                # # draw row label
                # row_lbl_font = pygame.font.SysFont('freesansbold.ttf', 20)

                # if col == 0:
                #     # make light grey font for dark brown tiles, dark grey font for light brown tiles
                #     color = (0, 0, 0)
                #     # create label
                #     row_lbl = row_lbl_font.render(str(ROWS - row), True, color)
                #     row_lbl_pos = (5, 5 + row * TILE)
                #     # draw lab
                #     surface.blit(row_lbl, row_lbl_pos)

                # # draw column label
                # col_lbl_font = pygame.font.SysFont('freesansbold.ttf', 20)
                # # make light grey font for dark brown tiles, dark grey font for light brown tiles
                # if row == 7:
                #     color = (0, 0, 0)
                #     # create label
                #     col_lbl = col_lbl_font.render(Tile.get_col_lett(col), True, color)
                #     col_lbl_pos = (col * TILE + TILE - 20, HEIGHT - 20)
                #     # draw label
                #     surface.blit(col_lbl, col_lbl_pos)


    def show_endgame(self, surface):
        '''
        This function shows the endgame screen once the game is over
        **Parameters**
            # surface: *pygame.Surface* - the surface to draw on
        **Returns**
            # none
        '''

        # draw a blue rectangle in the center of the screen
        color = (173, 216, 230) # light blue
        rect = (2 * TILE, 2 * TILE, 4 * TILE, 4 * TILE)
        pygame.draw.rect(surface, color, rect, 0, 64)
        pygame.draw.rect(surface, 'black', rect, 5, 64)
        
        # pass the gameover message to the screen
        font1 = pygame.font.Font('freesansbold.ttf', 32)
        text = font1.render('Gameover:', True, 'black', color)
        textRect = text.get_rect()
        textRect.center = (4 * TILE, 3 * TILE + TILE/2)
        
        # pass the winner to the screen
        surface.blit(text, textRect)
        text = font1.render(self.gameover, True, 'black', color)
        textRect = text.get_rect()
        textRect.center = (4 * TILE, 4 * TILE)
        surface.blit(text, textRect)
        
        # pass the restart message to the screen
        font2 = pygame.font.Font('freesansbold.ttf', 20)
        text = font2.render("Press 'r' to restart", True, 'black', color)
        textRect = text.get_rect()
        textRect.center = (4 * TILE, 4 * TILE + TILE/2)
        surface.blit(text, textRect)


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
        Counts the number of moves and halfmoves made
        After both players have made a move, the count will increment by 1
        Halfmoves are counted for moves in which a pawn is not moved or a piece is not captured
        Thus a pawn is moved or a piece is captured, the halfcount will reset to 0
        halfmoves is initialized to 0 at the start of the game
        fullmoves is initialized to 1 at the start of the game (as it only counts up when black moves)
        Function assists with the FEN notation, 50 move rule and stalemate

        **Parameters**
            # count: *float*, 0 at the start of the game
        **Returns**
            # count: *float*, the number of moves
        '''

        # counter for the number of moves made
        # increment the count by 0.5 for white moves
        # increment the count by 0.5 for black moves
        if self.player_color == 'white':
            if self.board.pawn_moved == False and self.board.piece_captured == False:
                self.halfmoves += 1
            else:
                self.halfmoves = 0
            self.fullmoves += 1
        else:
            if self.board.pawn_moved == False and self.board.piece_captured == False:
                self.halfmoves += 1
            else:
                self.halfmoves = 0


    def get_state(self):
        '''
        This method gets the state of the board for FEN notation
        **Parameters**
            # none
        **Returns**
            # none
        '''

        # sets an empty string to the board state
        self.board.state = ''
        
        # loop through the board and add the FEN notation to the board state
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

        # add the player turn
        if self.player_color == 'white':
            self.board.state += 'w'
        else:
            self.board.state += 'b'
        
        self.board.state += ' '

        # add castling
        no_castles = 0
        if self.board.K_castle == True:
            self.board.state += "K"
            no_castles += 1
        if self.board.Q_castle == True:
            self.board.state += "Q"
            no_castles += 1
        if self.board.k_castle == True:
            self.board.state += "k"
            no_castles += 1
        if self.board.q_castle == True:
            self.board.state += "q"
            no_castles += 1
        if no_castles == 0:
            self.board.state += '-'
        
        # create a dictionary to convert the tile number to a letter
        tile_dict = {
            1: "a",
            2: "b",
            3: "c",
            4: "d",
            5: "e",
            6: "f",
            7: "g",
            8: "h"
        }

        # check to see if en passant is possible
        if self.board.passant_tile != None:
            passant_square = tile_dict[self.board.passant_tile[1]+1] + str(8-self.board.passant_tile[0])
            self.board.state += " "
            self.board.state += passant_square
            self.board.passant_tile = None
        else:
            self.board.passant_tile = None
            self.board.state += " "
            self.board.state += "-"
            
        # add halfmoves
        self.board.state += ' '
        self.board.state += str(self.halfmoves)

        # add fullmoves
        self.board.state += ' '
        self.board.state += str(self.fullmoves)


    def state_tensor(self):
        '''
        This method converts the board state to a tensor
        **Parameters**
            # none
        **Returns**
            # none
        '''

        self.board.tensor = torch.zeros((6,8,8), dtype=torch.float)
        piece_dict = {
            'pawn': 0,
            'rook': 1,
            'knight': 2,
            'bishop': 3,
            'queen': 4,
            'king': 5
        }

        # loop through the board
        for row, _ in enumerate(self.board.tiles):
            for col, tile in enumerate(self.board.tiles[row]):
                if tile.piece_present():
                    if tile.piece.color == 'white':
                            self.board.tensor[piece_dict[tile.piece.name],row,col] = 1
                    elif tile.piece.color == 'black':
                            self.board.tensor[piece_dict[tile.piece.name],row,col] = -1


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


class Pawn(Piece):
    '''
    This class represents a pawn, which is a child of the Piece class
    **Attributes**
        # dir - the direction the pawn can move
        # en_passant - whether or not the pawn can be taken en passant
        # FEN_notation - the FEN notation for the pawn
        # inherited from Piece (name, color, moves, moved, img, img_rect)
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
        # FEN_notation: *str*, the FEN notation of the knight
        # inherited from Piece (name, color, moves, moved, img, img_rect)
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
        # FEN_notation, *str* - the FEN notation of the bishop
        # inherited from Piece (name, color, moves, moved, img, img_rect)
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
        # FEN_notation: *str*, the FEN notation of the rook
        # inherited from Piece (name, color, moves, moved, img, img_rect)
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
        # FEN_notation, *str* the FEN notation of the queen
        # inherited from Piece (name, color, moves, moved, img, img_rect)
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
        # FEN_notation: *str*, the FEN notation of the king
        # l_rook: *Rook obj*, the left rook of the king
        # r_rook: *Rook obj*, the right rook of the king
        # inherited from Piece (name, color, moves, moved, img, img_rect)
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
        # __eq__: *bool*, checks to see if two tiles are equal
        # piece_present: *bool*, checks to see if a piece is present on the tile
        # empty_tile: *bool*, checks to see if the tile is empty
        # moveable_square: *bool*, checks to see if the tile is a moveable square
        # enemy_present: *bool*, checks to see if an enemy piece is present on the tile
        # friendly_present: *bool*, checks to see if a friendly piece is present on the tile
        # get_col_lett: *str*, returns the column letter of the tile
    '''

    # dictionary of column letters
    COL_LETT = {0: 'a', 
                1: 'b', 
                2: 'c', 
                3: 'd', 
                4: 'e',
                5: 'f',
                6: 'g',
                7: 'h'
                }

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
    

    @staticmethod
    def get_col_lett(col):
        '''
        This function returns the column letter of a given column number
        **Parameters**
            # col: *int*, the column number
        **Returns**
            # *str*, the column letter
        '''

        return Tile.COL_LETT[col]


class Board:
    '''
    This class represents the board
    **Attributes**
        # tiles: *list*, a 2D list of Tile objects
        # state: *str*, the state of the board (in play, checkmate, stalemate)
        # tensor: *torch tensor*, a tensor representation of the board
        # white_moves: *list*, a list of all possible white moves
        # white_moves_extra: *list*, a list of all possible white moves (including castling)
        # white_moves_bot: *list*, a list of all possible white moves (including castling and en passant)
        # white_moves_len: *int*, the length of the white_moves list
        # black_moves: *list*, a list of all possible black moves
        # black_moves_extra: *list*, a list of all possible black moves (including castling)
        # black_moves_bot: *list*, a list of all possible black moves (including castling and en passant)
        # black_moves_len: *int*, the length of the black_moves list
        # passant_tile: *Tile obj*, the tile that can be captured en passant
        # K_castle: *bool*, True if white can castle kingside, False otherwise
        # Q_castle: *bool*, True if white can castle queenside, False otherwise
        # k_castle: *bool*, True if black can castle kingside, False otherwise
        # q_castle: *bool*, True if black can castle queenside, False otherwise
        # pawn_moved: *bool*, True if a pawn has moved, False otherwise
        # piece_captured: *bool*, True if a piece has been captured, False otherwise
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
        self.white_moves_extra = []
        self.white_moves_bot = []
        self.white_moves_len = 0
        self.black_moves = []
        self.black_moves_extra = []
        self.black_moves_bot = []
        self.black_moves_len = 0
        self.passant_tile = None
        self.K_castle = True
        self.Q_castle = True
        self.k_castle = True
        self.q_castle = True
        self.pawn_moved = False
        self.piece_captured = False

    
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

        # set the initial and final tiles
        initial = move.init_tile
        final = move.final_tile

        # check to see if a piece is captured
        if self.tiles[final.row][final.col].enemy_present(piece.color):
            self.piece_captured = True
        else:
            self.piece_captured = False

        # used to preset an en passant tile
        en_passant_tile = self.tiles[final.row][final.col].empty_tile()

        # update the board, remove the initial piece
        self.tiles[initial.row][initial.col].piece = None
        # add the piece to the final tile
        self.tiles[final.row][final.col].piece = piece

        # methods for special pawn movements
        if isinstance(piece, Pawn):
            # check to see if the pawn has moved
            self.pawn_moved = True
            
            # method for en passant
            diff = final.col - initial.col
            if diff != 0 and en_passant_tile:
                self.tiles[initial.row][initial.col + diff].piece = None
                self.tiles[final.row][final.col].piece = piece

            # method for pawn promotion
            elif final.row == 0 or final.row == 7:
                self.tiles[final.row][final.col].piece = Queen(piece.color)
        else:
            self.pawn_moved = False

        # method for castling
        if isinstance(piece, King):
            if self.castle(initial, final) and not testing:
                # check to see what direction the king is castling
                diff = final.col - initial.col
                if diff < 0:
                    # castle left
                    if piece.color == 'white':
                        init_tile = Tile(7, 0, piece.l_rook)
                        final_tile = Tile(7, 3, piece.l_rook)
                        move = Move(init_tile, final_tile)
                        self.move_piece(piece.l_rook, move)
                    if piece.color == 'black':
                        init_tile = Tile(0, 0, piece.l_rook)
                        final_tile = Tile(0, 3, piece.l_rook)
                        move = Move(init_tile, final_tile)
                        self.move_piece(piece.l_rook, move)
                else:
                    # castle right
                    if piece.color == 'white':
                        init_tile = Tile(7, 7, piece.r_rook)
                        final_tile = Tile(7, 5, piece.r_rook)
                        move = Move(init_tile, final_tile)
                        self.move_piece(piece.r_rook, move)
                    if piece.color == 'black':
                        init_tile = Tile(0, 7, piece.r_rook)
                        final_tile = Tile(0, 5, piece.r_rook)
                        move = Move(init_tile, final_tile)
                        self.move_piece(piece.r_rook, move)
            
            # set FEN notations for castling
            if piece.color == 'white':
                self.K_castle = False
                self.Q_castle = False
            elif piece.color == 'black':
                self.k_castle = False
                self.q_castle = False
                    
        if isinstance(piece, Rook):
            if initial.col == 0:
                if piece.color == 'white':
                    self.Q_castle = False
                elif piece.color == 'black':
                    self.q_castle = False
            elif initial.col == 7:
                if piece.color == 'white':
                    self.K_castle = False
                elif piece.color == 'black':
                    self.k_castle = False

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

        # check to see if the king is castling (only time king can move 2 tiles)
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
            # leave the method if the piece is not a pawn
            return
        
        for row in range(ROWS):
            for col in range(COLS):
                if isinstance(self.tiles[row][col].piece, Pawn):
                    # checks to see if a piece is occupying the en passant tile
                    self.tiles[row][col].piece.en_passant = False

        piece.en_passant = True


    def move_king_check(self, piece, move):
        '''
        Checks to see if the king is in check by creating a board copy and moving the piece
        **Parameters**
            # piece: *Piece obj*, the piece to move
            # move: *Move obj*, the move to make
        **Returns**
            # *bool*: True if the king is in check, False otherwise
        '''

        # create a board copy
        board_copy = copy.deepcopy(self)
        piece_copy = copy.deepcopy(piece)
        final_row = move.final_tile.row
        final_col = move.final_tile.col
        # move the piece on the board copy
        board_copy.move_piece(piece_copy, move, testing=True)

        if isinstance(piece, King):
            for i in range(final_row - 1, final_row + 2):
                for j in range(final_col - 1, final_col + 2):
                    if Tile.on_board(i,j) and board_copy.tiles[i][j].enemy_present(piece.color):
                        if board_copy.tiles[i][j].piece.name == 'king':
                            return True
        
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


    def is_king_check(self, color):
        '''
        Function to check if the king is in check
        **Parameters**
            # color: *str*, the color of the king to check
        **Returns**
            # *bool*: True if the king is in check, False otherwise
        '''

        # check to see if white king is in check
        if color == 'white':
            self.all_valid_moves('black')
            for move in self.black_moves:
                row = move.final_tile.row
                col = move.final_tile.col

                if self.tiles[row][col].piece_present() and self.tiles[row][col].piece.name == 'king':
                    return True
            return False

        # check to see if black king is in check
        if color == 'black':
            self.all_valid_moves('white')
            for move in self.white_moves:
                row = move.final_tile.row
                col = move.final_tile.col

                if self.tiles[row][col].piece_present() and self.tiles[row][col].piece.name == 'king':
                    return True
            return False
        

    def is_checkmate(self, color):
        '''
        Function to check if the king is in checkmate
        **Parameters**
            # color: *str*, the color of the king to check
        **Returns**
            # *bool*: True if the king is in checkmate, False otherwise
        '''

        # check to see if black king is in checkmate
        if color == 'black':
            self.all_valid_moves('black')
            # check to see if no valid moves and king is in check
            if self.black_moves == [] and self.is_king_check('black'):
                return True
            return False
        
        # check to see if white king is in checkmate
        if color == 'white':
            self.all_valid_moves('white')
            # check to see if no valid moves and king is in check
            if self.white_moves == [] and self.is_king_check('white'):
                return True
            return False
        

    def valid_moves(self, piece, row, col, k_check=True):
        '''
        Generates the valid moves for a piece
        contains subfunctions:
            # iterate_moves(directions) - iterates through the directions of a piece for pieces that move in a straight line
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
                                    if not self.move_king_check(piece, move):
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
                                    if not self.move_king_check(piece, move):
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

            # add moves for forward movementc
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
                            if not self.move_king_check(piece, move):
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
                        chosen = Tile(poss_row, i, final_piece)
                        move = Move(init_tile, chosen)

                        # check to see if the king is in check
                        if k_check:
                            if not self.move_king_check(piece, move):
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
                            self.passant_tile = (final_tile.row, final_tile.col)
                            move = Move(init_tile, final_tile)
                        
                            # check to see if the king is in check
                            if k_check:
                                if not self.move_king_check(piece, move):
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
                            self.passant_tile = (final_tile.row, final_tile.col)
                            move = Move(init_tile, final_tile)
                            
                            # check to see if the king is in check
                            if k_check:
                                if not self.move_king_check(piece, move):
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
                            if not self.move_king_check(piece, move):
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
                            if not self.move_king_check(piece, move):
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
                                    if not self.move_king_check(piece, move_king) and not self.move_king_check(l_rook, move_rook):
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
                                    if not self.move_king_check(piece, move_king) and not self.move_king_check(r_rook, move_rook):
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
        '''
        Finds all the valid moves for a given color
        **Parameters**
            color: *str*
                The color of the pieces to find the valid moves for (white or black)
        **Returns**
            None
        '''

        # create empty lists for the white and black pieces
        self.white_moves = []
        self.black_moves = []
        self.white_moves_bot = []
        self.black_moves_bot = []
        
        # iterate through the board and append valid moves to the lists
        for row, _ in enumerate(self.tiles):
            for col, tile in enumerate(self.tiles[row]):
                if tile.piece_present() and tile.piece.color == color:
                    piece = tile.piece
                    self.valid_moves(piece, row, col)
                    for move in piece.moves:
                        if color == 'white':
                            self.white_moves.append(move)
                        elif color == 'black':
                            self.black_moves.append(move)
                        piece.clear_moves()
        
        # get the length of the list of moves
        if color == 'white':
            self.white_moves_len = len(self.white_moves)
        elif color == 'black':
            self.black_moves_len = len(self.black_moves)

        self.white_moves_bot = self.white_moves.copy()
        self.black_moves_bot = self.black_moves.copy()

        # max number of moves possible in a single position is 218
        if len(self.white_moves_bot) != 0 or len(self.black_moves_bot) != 0:
            while len(self.white_moves_bot) != 218 and len(self.black_moves_bot) != 218:
                if len(self.white_moves_bot) != 0:
                    self.white_moves_bot.append(((0,0),(0,0)))
                elif len(self.black_moves_bot) != 0:
                    self.black_moves_bot.append(((0,0),(0,0)))


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
    '''
    Class to handle the chess AI
    child of the nn.Module class from PyTorch
    **Attributes**
        conv1: *nn.Conv2d*, the first convolutional layer
        conv2: *nn.Conv2d*, the second convolutional layer
        conv3: *nn.Conv2d*, the third convolutional layer
        fc1: *nn.Linear*, the first fully connected layer, using a linear activation function
        fc2: *nn.Linear*, the second fully connected layer, using a linear activation function
        file: *str*, the file to save the model to
    **Methods**
        __init__: constructor for the ChessAI class
        forward: forward pass of the neural network
        save_model: saves the model to a file
        load_model: loads the model from a file
    '''

    def __init__(self, model_file=None, num_actions=218):
        '''
        Method to initialize the ChessAI class
        **Parameters**
            model_file: *str*, the file to save the model to
            num_actions: *int*, the number of actions the AI can take
        '''
        super(ChessAI, self).__init__()
        # initialize the layers, in_channels is 6 because there are 6 channels in the state tensor
        # out_channels is the number of filters
        # kernel_size is the size of the kernel
        # padding is the amount of padding to add to the input
        # nn.Conv2d is the convolutional layer
        self.conv1 = nn.Conv2d(in_channels=6, out_channels=32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1)
        self.fc1 = nn.Linear(in_features=128 * 8 * 8, out_features=256)
        self.fc2 = nn.Linear(in_features=256, out_features=num_actions)

        self.file = model_file


    def forward(self, state_tensor, num_moves):
        '''
        Method to perform a forward pass of the neural network
        **Parameters**
            state_tensor: *torch.Tensor*, the state tensor of the board
            num_moves: *int*, the number of possible moves
        **Returns**
            *torch.Tensor*, the output tensor of the neural network
        '''

        state_tensor = state_tensor.unsqueeze(0)
        # use the ReLU activation function for the convolutional layers
        state_tensor = F.relu(self.conv1(state_tensor))
        state_tensor = F.relu(self.conv2(state_tensor))
        state_tensor = F.relu(self.conv3(state_tensor))

        # flatten the tensor
        state_tensor = state_tensor.view(-1, 128 * 8 * 8)

        # use the ReLU activation function for the fully connected layers
        state_tensor = F.relu(self.fc1(state_tensor))
        state_tensor = self.fc2(state_tensor)

        # Resize output tensor to match the number of possible moves
        state_tensor = state_tensor[:, :num_moves]

        # Return softmax of output tensor
        return F.softmax(state_tensor, dim=1)
    

    def save_model(self):
        '''
        Function to save the model to a file
        **Parameters**
            None
        **Returns**
            None
        '''
        return torch.save(self.state_dict(), 'ChessAI.pt')


    def load_model(self):
        '''
        Funcion to load the model from a file
        **Parameters**
            None
        **Returns**
            None
        '''

        return self.load_state_dict(torch.load('ChessAI.pt'))
    

class ChessAgent():
    '''
    Class to handle the chess agent, which uses the ChessAI class
    **Attributes**
        model: *ChessAI*, the model to use
        optimizer: *optim.Adam*, the optimizer to use
        gamma: *float*, the discount factor
    **Methods**
        __init__: constructor for the ChessAgent class
        select_action: selects an action to take
        update_model: updates the model
    '''

    def __init__(self, model, learning_rate=0.001, gamma=0.99):
        '''
        Constructor for the ChessAgent class
        **Parameters**
            model: *ChessAI*, the model to use
            learning_rate: *float*, the learning rate to use
            gamma: *float*, the discount factor
        **Returns**
            None
        '''

        self.model = model
        # use the Adam optimizer
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        self.gamma = gamma
    

    def select_action(self, state_tensor, num_moves, noise_scale=0.0):
        '''
        Method to select an action to take
        **Parameters**
            state_tensor: *torch.Tensor*, the state tensor of the board
            num_moves: *int*, the number of possible moves
            noise_scale: *float*, the amount of noise to add to the action
        **Returns**
            action: *int*, the action to take
            state_value: *torch.Tensor*, the state value of the board
        '''

        # get the logits and add noise
        # logits are the output of the neural network before the softmax activation function
        # noise is used to add some randomness to the action
        logits = self.model.forward(state_tensor, num_moves)
        noise = noise_scale * torch.randn_like(logits)

        # get the action and state value
        # the action is the index of the maximum value of the logits
        # the state value is the softmax of the logits
        probs = F.softmax(logits + noise, dim=-1)
        action = torch.multinomial(probs, 1).item()
        state_value = probs[0, action]
        return action, state_value


    def update_model(self, state_value, reward, next_state_tensor, next_num_moves, gameover):
        '''
        Method to update the model following each action
        **Parameters**
            state_value: *torch.Tensor*, the state value of the board
            reward: *float*, the reward for the action
            next_state_tensor: *torch.Tensor*, the state tensor of the next board
            next_num_moves: *int*, the number of possible moves for the next board
            gameover: *bool*, whether the game is over
        **Returns**
            None
        '''

        # set the gradients to zero
        # gradients are used to update the weights of the neural network
        self.optimizer.zero_grad()

        # calculate the next state value
        next_state_value = torch.zeros(1, 1)
        
        # use the model to get the next state value if the game is not over
        if not gameover:
            next_state_value = self.model.forward(next_state_tensor, next_num_moves).max(1)[0].detach().unsqueeze(1)
        expected_state_value = reward + self.gamma * next_state_value
        loss = (state_value - expected_state_value).pow(2).mean()
        
        # backpropagate the loss
        # backpropagation is used to update the weights of the neural network
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()


if __name__ == '__main__':
    train_or_play = input("Train or Play: ")
    if train_or_play == "Play":
        type_of_game = input("What type of game? PVP or Bot: ")
        Chess_App = Chess_App()
        Chess_App.run(type_of_game)
    else:
        # assert that the OS is Windows
        if (platform.system() != 'Windows'):
            print("ATTENTION: This model was trained on Windows, so it can only be trained in a Windows Environment")
        else:
            Chess_App = Chess_App()
            Chess_App.train()
