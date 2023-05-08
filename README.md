# Chess
A game where you can play chess against another human or bot, or train the bot against itself.

## 1. Introduction to Chess
Chess is a game that has 64 squares, where each color has 16 pieces, white and black. White always moves first. There are three possilbe conclusions white wins, black wins, or draw. In this game there are two modes for playing. The first is two humans and the second is one human and one bot. Another feature of this game is the ability to train the bot. During this state two bots play against each other indefinetely to train the model.

## 2. Requirements
Please see the requirements.txt file to ensure your environment is set up correctly.

Note: the AI has been trained in stockfish using a .exe file that can only be run on a PC.
Running the main file (as described below), will run in game mode automatically. Instructions below describe changing to training mode.

## 3. Starting the game
To run the game you have to run the Chess.py file. Inside this file all lines beside the .train() must be uncommented.
![image](https://user-images.githubusercontent.com/121264060/236707930-95f72866-8521-46d0-90c8-a72682f06eec.png)

Once this runs there will be a message in the terminal. Type either "PVP" or "Bot" depending on the mode you wish to play.
![image](https://user-images.githubusercontent.com/121264060/236708017-5f7187a5-35f6-4060-a721-0a04fb0721e0.png)

If you wish to train the bot uncomment the train() function and comment out the run() function and the input() function.
![image](https://user-images.githubusercontent.com/121264060/236708064-9353d5c0-050f-4568-a4eb-e3b1f90122a7.png)

## 3. Gameplay
### Playing the Game
To play the game you have to click and drag the pieces on your respective turn. Once a piece is clicked the squares where a legal move is possible will become highlighted. Different colors of highlight have no meaning other than being a light or dark square where the piece could move. If you would like to restart at any point in the game press "r".

### Completion of the Game
If the game is over a blue screen will appear to denote the end of the game. If you would like to restart press "r". If you would like to exit the game press the red X key.
![image](https://user-images.githubusercontent.com/121264060/236708358-fa6c3cbe-5306-4036-9b54-0d3e45378e24.png)


## 4. Explanation of the Code
The entire game is written in python with Classes. For training the bot Stockfish is used, this file is located within the repository. This is only used to calculate the evaluation of the position for providing a reward while training the bot.
