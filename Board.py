'''
This code will create the gui for the board and the pieces
players will be able to interact with the board and pieces
pygame will be used to help make the gui
'''

import pygame as py
py.init()

# this class will help set the tiles of the board
# white side of the board should have a1 at the bottom left
# a1 should be black
class Tile:
    def __init__(self, x, y, width, height):
        self.x = x
        self.y = y
        self.width = width
        self.height = height

        # even squares are white, odd squares are black
        if (x + y) % 2 == 0:
            self.colour = 'white'
        else: 
            self.colour = 'black'
        
        # set the colour of the tile
        if self.colour == 'white':
            self.draw_colour = (247, 207, 164)
        else:
            self.draw_colour = (199, 141, 83)

        self.piece = None
        self.tile_x = x * width
        self.tile_y = y * height
        self.tile_pos = (self.tile_x, self.tile_y)
        self.move_colour = (97, 235, 110)
        self.attack_colour = (230, 47, 73)
        self.highlight_move = False
        self.highlight_attack = False
        self.piece = None
        self.rect = py.Rect(self.tile_x, self.tile_y, self.width, self.height)

    def get_pos(self):
        columns = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H']
        return columns[self.x] + str(self.y + 1)
    
    def highlight(self, display):
        # set tile colours
        if self.highlight_move:
            py.draw.rect(display, self.move_colour, self.rect)

        elif self.highlight_attack:
            py.draw.rect(display, self.attack_colour, self.rect)

        else:
            py.draw.rect(display, self.colour, self.rect)
        
        # place the piece in the centre of the tile
        if self.piece != None:
            center_rect = self.piece.img.get_rect()
            center_rect.center = self.rect.center
            display.blit(self.piece.img, center_rect)

class Board:
    def __init__(self, width, height):
        self.width = width
        self.height = height
        # ensure that the board fits equally into the window
        self.tile_width = width // 8
        self.tile_height = height // 8
        self.selected_tile = None
        # first turn starts with white
        self.turn = 'white'
        # set the inital board configuration
        self.config = [
            ['bR', 'bN', 'bB', 'bQ', 'bK', 'bB', 'bN', 'bR'],
            ['bP', 'bP', 'bP', 'bP', 'bP', 'bP', 'bP', 'bP'],
            ['', '', '', '', '', '', '', ''],
            ['', '', '', '', '', '', '', ''],
            ['', '', '', '', '', '', '', ''],
            ['', '', '', '', '', '', '', ''],
            ['wP', 'wP', 'wP', 'wP', 'wP', 'wP', 'wP', 'wP'],
            ['wR', 'wN', 'wB', 'wQ', 'wK', 'wB', 'wN', 'wR']
        ]
        self.tiles = self.create_tiles()
        # self.setup_board()

    # create the tiles for the board using the Tile class
    def create_tiles(self):
        setup = []
        for i in range(8):
            for j in range(8):
                setup.append(Tile(i, j, self.tile_width, self.tile_height))
        return setup
    
    # retrieve the tile at a given position
    def get_tile(self, position):
        for tile in self.tiles:
            if tile.get_pos(tile.x, tile.y) == (position):
                return tile

    # retrieve a piece at a given position
    def get_piece(self, position):
        return self.get_tile(position).piece
    
    # create a function that handles user input
    # i and j are the x and y coordinates of the click
    def user_input(self, i, j):
        x = i // self.tile_width
        y = j // self.tile_height
        clicked_tile = self.get_tile((x, y))
        # if the user clicks on a tile with a piece, select that piece if it is the correct turn
        if clicked_tile.piece != None:
            if clicked_tile.piece.colour == self.turn:
                self.selected_piece = clicked_tile.piece.colour
        elif clicked_tile.piece == None:
            if self.selected_piece != None:
                self.move_piece(self.selected_piece, clicked_tile)
                self.selected_piece = None
        # if the user moves a piece, switch the turn
        elif self.selected_piece.move(self, clicked_tile):
            self.turn = 'black' if self.turn == 'white' else 'white'

    # create a function that highlights possible moves of a piece
    def highlight(self, display):
        if self.selected_piece != None:
            self.get_tile(self.selected_piece.position).highlight_move = True
            for tile in self.selected_piece.get_valid_moves(self):
                tile.highlight_move = True
            for tile in self.selected_piece.get_valid_attacks(self):
                tile.highlight_attack = True
        for tile in self.tiles:
            tile.draw(display)

if __name__ == "__main__":
    board = Board(800, 800)
    display = py.display.set_mode((board.width, board.height))

    running = True
    while running:
        for event in py.event.get():
            if event.type == py.QUIT:
                running = False
        
        for tile in board.tiles:
            tile.highlight(display)

        py.display.update()

    py.quit()