# Chess
A game where you can play chess against another human or bot, or train the bot against itself.

## 1. Introduction to Chess
Chess is a game that has 64 squares, where each color has 16 pieces, white and black. White always moves first. There are three possilbe conclusions white wins, black wins, or draw. In this game there are two modes for playing. The first is two humans and the second is one human and one bot. Another feature of this game is the ability to train the bot. During this state two bots play against each other indefinetely to train the model.

## 2. Requirements
Please see the requirements.txt file to ensure your environment is set up correctly.

<strong>Note: the AI has been trained in stockfish using a .exe file that can only be run on a Windows OS environment.</strong>

Running the main file (as described below), will run in game mode automatically. Instructions below describe changing to training mode.

## 3. Installation

1. Clone this repository to your local machine.

```sh
git clone https://github.com/Gaulk/Chess.git
```

2. Change the directory to the project folder.

3. Create a virtual environment and activate it.

```sh
python -m venv venv
source venv/bin/activate
```

4. Install the required packages.

```sh
pip install -r requirements.txt
```

## 3. Starting the game
To run the game you have to run the Chess.py file. Inside this file all lines beside the .train() must be uncommented.
```py
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
 ```

Once this runs there will be a message in the terminal:
```
Hello from the pygame community. https://www.pygame.org/contribute.html
Train or Play:
```

Type ```Play``` to start the main game mode.

You will be prompted with the following:

```What type of game? PVP or Bot:```

Type ```PVP``` if you wish to play against a friend.
Type ```Bot``` if you wish to play against our reinforcement ML trained bot.

If you wish to train the bot Type ```Train``` at the first user input. 
NOTE: this function will only be available if you are running a Windows environment.

## 3. Gameplay
### Playing the Game
To play the game you have to click and drag the pieces on your respective turn. Once a piece is clicked the squares where a legal move is possible will become highlighted. Different colors of highlight have no meaning other than being a light or dark square where the piece could move. If you would like to restart at any point in the game press "r".

### Completion of the Game
If the game is over a blue screen will appear to denote the end of the game. If you would like to restart press "r". If you would like to exit the game press the red X key.
![image](https://user-images.githubusercontent.com/121264060/236708358-fa6c3cbe-5306-4036-9b54-0d3e45378e24.png)


## 4. Explanation of the Code
The entire game is written in python with Classes. For training the bot Stockfish is used, this file is located within the repository. This is only used to calculate the evaluation of the position for providing a reward while training the bot.
