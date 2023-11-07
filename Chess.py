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

from kivy.app import App
from kivy.uix.gridlayout import GridLayout
from kivy.uix.floatlayout import FloatLayout
from kivy.uix.button import Button
from kivy.uix.label import Label
from kivy.uix.image import Image
from kivy.core.window import Window
from kivy.properties import NumericProperty
from kivy.uix.boxlayout import BoxLayout
from kivy.lang import Builder
from kivy.uix.widget import Widget
from kivy.properties import ListProperty
from kivy.uix.behaviors import DragBehavior

Builder.load_file("Chess.kv")

class Chess(App):
    def build(self):
        # app = FloatLayout()
        centered_chess = Chessboard()
        # centered_chess.pos_hint = {'center_x': 0.5, 'center_y': 0.5}
        # app.add_widget(centered_chess)
        centered_chess.add_starting_pieces(centered_chess.initial_pieces)
        return centered_chess

class DraggableChessPiece(DragBehavior, Image):
    def __init__(self, piece_image, **kwargs):
        super(DraggableChessPiece, self).__init__(**kwargs)
        self.source = piece_image
        self.touch_offset = (0,0)
        # self.current_parent = None
        self.old_tile = self.parent
        self.current_position_col = None
        self.current_position_row = None

    # def on_touch_down(self, touch):
    #     if self.collide_point(*touch.pos):
    #         # print(self.parent)
    #         self.old_tile = self.parent
    #         print(self.old_tile)

    #         self.touch_offset = (self.x - touch.x, self.y - touch.y)
    #         # print(self.touch_offset)
    #         # self.current_parent = self.parent
    #         return True

    # def on_touch_move(self, touch):
    #     if self.touch_offset != (0, 0):
    #         # Use these two to toggle if can drag or not
    #         # This moves the piece itself
    #         self.x = touch.x + self.touch_offset[0]
    #         self.y = touch.y + self.touch_offset[1]

    #         # Calculate where the piece is on the board when dragging
    #         # self.current_position_col = int(8 - ((touch.x//125) + 1))
    #         # self.current_position_row = int(touch.y//95)
    #         # print(self.current_position_row)
    #         # print(self.current_position_col)
    #         # print(f"{position_row}, {position_col}")
            
    #         # new_parent = self.parent
    #         # if new_parent != self.current_parent:
    #         #     if new_parent:
    #         #         if self.current_parent:
    #         #             self.current_parent.remove_widget(self)  # Remove from the old parent
    #         #         new_parent.add_widget(self)  # Add to the new parent
    #         #     else:
    #         #         self.current_parent.add_widget(self)  # Restore to the old parent

    # def on_touch_up(self, touch):
    #     # Calculate where the piece is when dropped
    #     self.current_position_col = int(8 - ((touch.x//125) + 1))
    #     self.current_position_row = int(touch.y//95)
    #     print(self.old_tile)
    #     if self.old_tile == None:
    #         self.old_tile = self.parent
    #     print(self.old_tile)
        
    #     # self.old_tile.remove_piece_from_tile(self)
    #     # app = App.get_running_app()
    #     # board = app.root
    #     # new_tile = board.children[board.select_tile(self.current_position_row,self.current_position_col)]
    #     # new_tile.add_widget(self)
    #     self.touch_offset = (0,0)

    # Override the on_touch_down method to detect when the user touches the widget
    def on_touch_down(self, touch):
        if self.collide_point(*touch.pos):
            # If the touch event occurred within the widget's bounds, handle the touch event
            # by setting the widget as the current touch target
            touch.grab(self)
            # Add what the tile the piece was originally on
            self.old_tile = self.parent
            
            return True
        return super().on_touch_down(touch)

    # Override the on_touch_move method to track the movement of the user's finger
    def on_touch_move(self, touch):
        if touch.grab_current == self:
            # If the touch event is being handled by our widget, update the widget's position
            self.pos = (self.pos[0] + touch.dx, self.pos[1] + touch.dy)
            print(self.pos)
            # Calculate what row and col the piece is currently on
            self.current_position_col = int(8 - ((self.pos[0]//125) + 1))
            self.current_position_row = int(self.pos[1]//95)
            # print(f"Row:{self.current_position_row}, Col:{self.current_position_col}")

    # Override the on_touch_up method to update the widget's position when the touch event ends
    def on_touch_up(self, touch):
        if touch.grab_current == self:
            # If the touch event is being handled by our widget, release the widget as the current
            # touch target and handle the touch event
            touch.ungrab(self)
            print("Touch Up")
            print(self.old_tile)
            print(f"Row:{self.current_position_row}, Col:{self.current_position_col}")

            # Remove piece from previous tile and add it to where it was dragged to
            self.old_tile.remove_piece_from_tile(self)
            app = App.get_running_app()
            board = app.root
            new_tile = board.children[board.select_tile(self.current_position_row,self.current_position_col)]
            new_tile.add_widget(self)
            return True
        return super().on_touch_up(touch)

class Tile(GridLayout):
    background_color = ListProperty([1, 1, 1, 1])
    ## Use this to change the background color when highlighting legal moves
    # tile.background_color = ((255/255.0, 0/255.0, 0/255.0, 1))

    def __init__(self, piece=None, **kwargs):
        super(Tile, self).__init__(**kwargs)
        self.piece = piece
        self.rows = 1
        self.cols = 1
        self.spacing = 0
        # self.size = (50, 50)
        # self.size_hint = (1, 1)
        self.touch_offset = (0,0)
        self.current_parent = None
        self.old_tile = None
        self.current_position_col = None
        self.current_position_row = None
        
    def add_piece_to_tile(self, piece):
        self.piece = piece
        
        # Set the position of the piece
        self.piece.size_hint = (1, 1)
        self.piece.pos_hint = {'center_x': 0.5, 'center_y': 0.5}

        self.add_widget(self.piece)

    def remove_piece_from_tile(self, piece):
        self.piece = None
        self.remove_widget(piece)

class Chessboard(GridLayout):
    def __init__(self, **kwargs):
        super(Chessboard, self).__init__(**kwargs)
        self.rows = 8
        self.cols = 8
        self.spacing = 0
        self.size_hint = (1, 1)

        # Define piece images
        ## Add each starting piece, then create an item for each in the loop
        self.initial_pieces = {
            # White Pieces
            'WhitePawn_A2': ['wpawn.png', (1,7)],
            'WhitePawn_B2': ['wpawn.png', (1,6)],
            'WhitePawn_C2': ['wpawn.png', (1,5)],
            'WhitePawn_D2': ['wpawn.png', (1,4)],
            'WhitePawn_E2': ['wpawn.png', (1,3)],
            'WhitePawn_F2': ['wpawn.png', (1,2)],
            'WhitePawn_G2': ['wpawn.png', (1,1)],
            'WhitePawn_H2': ['wpawn.png', (1,0)],
            'WhiteRook_A1': ['wrook.png', (0,7)],
            'WhiteRook_H1': ['wrook.png', (0,0)],
            'WhiteKnight_B1': ['wknight.png', (0,6)],
            'WhiteKnight_G1': ['wknight.png', (0,1)],
            'WhiteBishop_C1': ['wbishop.png', (0,5)],
            'WhiteBishop_F1': ['wbishop.png', (0,2)],
            'WhiteQueen_D1': ['wqueen.png', (0,4)],
            'WhiteKing_E1': ['wking.png', (0,3)],
            # Black Pieces
            'BlackPawn_A7': ['bpawn.png', (6,7)],
            'BlackPawn_B7': ['bpawn.png', (6,6)],
            'BlackPawn_C7': ['bpawn.png', (6,5)],
            'BlackPawn_D7': ['bpawn.png', (6,4)],
            'BlackPawn_E7': ['bpawn.png', (6,3)],
            'BlackPawn_F7': ['bpawn.png', (6,2)],
            'BlackPawn_G7': ['bpawn.png', (6,1)],
            'BlackPawn_H7': ['bpawn.png', (6,0)],
            'BlackRook_A8': ['brook.png', (7,7)],
            'BlackRook_H8': ['brook.png', (7,0)],
            'BlackKnight_B8': ['bknight.png', (7,6)],
            'BlackKnight_G8': ['bknight.png', (7,1)],
            'BlackBishop_C8': ['bbishop.png', (7,5)],
            'BlackBishop_F8': ['bbishop.png', (7,2)],
            'BlackQueen_D8': ['bqueen.png', (7,4)],
            'BlackKing_E8': ['bking.png', (7,3)],
        }

        # Create board
        for i in range(8):
            for j in range(8):
                # Set colors for dark and light squares
                color = (200/255.0, 141/255.0, 83/255.0, 1) if (i + j) % 2 != 0 else (246/255.0, 207/255.0, 164/255.0, 1)
                # Create each tile
                tile = Tile(background_color=color)

                # Add the widget to the board
                self.add_widget(tile)

    def add_starting_pieces(self, pieces):
        for piece in pieces:
            piece_item = DraggableChessPiece(piece_image=pieces[piece][0])
            tile = self.children[self.select_tile(pieces[piece][1][0],pieces[piece][1][1])]
            tile.add_piece_to_tile(piece_item)

    def select_tile(self, row, col):
        return row*8 + col

if __name__ == '__main__':
    Chess().run()
