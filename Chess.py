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
from kivy.uix.widget import Widget
from kivy.lang import Builder


