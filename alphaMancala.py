import os
import re
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import pickle
from gameEnvironment import PLAYER_ONE_PITS, PLAYER_TWO_PITS, PLAYER_ONE, PLAYER_TWO, HISTORY_LENGTH, FUTURE_TURNS_TO_PREDICT, EXPORT_FOLDER, EXPORT_FILENAME, EXPORT_LOSSES_FILENAME
from typing import TYPE_CHECKING
from IPython import get_ipython
import sys

if TYPE_CHECKING:
    from mancalaGUI import MancalaGUI


class AlphaMancala():
    '''
    This class represents the mancala 
    agent which interacts with the game.
    '''
    
    def __init__(self, player_number, is_training, model_number_to_use=None) -> None:
        '''
        Initialize my deep learning model to play from a certain player's 
        perspective.
        '''

        self.player_number: int = player_number
        self.is_training = is_training

        self.model = AlphaMancalaMind(model_number_to_use=model_number_to_use)
            

    def generate_illegal_moves_mask(self: 'AlphaMancala', board: list[int]) -> torch.tensor:
        '''
        Given a board state, this function generates a mask where 
        legal moves are marked with a 1 and illegal moves are marked with
        a 0.
        '''

        if self.player_number == PLAYER_ONE:
            return torch.tensor([1 if board[pit] != 0 else 0 for pit in PLAYER_ONE_PITS], dtype=torch.bool)

        elif self.player_number == PLAYER_TWO:
            return torch.tensor([1 if board[pit] != 0 else 0 for pit in PLAYER_TWO_PITS], dtype=torch.bool)
        
        else:
            raise ValueError(f"Invalid player number: {self.player_number}")
        

    def get_future_values(self, current_state, num_turns_in_advance):

        # where in the state is the player turn flag
        player_index = len(PLAYER_ONE_PITS) + len(PLAYER_TWO_PITS) + 2

        # who is the current player according to the state?
        current_player = PLAYER_ONE if current_state[player_index] < 0.5 else PLAYER_TWO
        current_players_pits = PLAYER_ONE_PITS if current_state[player_index] < 0.5 else PLAYER_TWO_PITS

        # init to -inf for all choices
        action_values = [float('-inf')] * len(current_players_pits)

        if num_turns_in_advance > 1:

            for pit in current_players_pits:
                # for each choice in pit
                policy_logits, value, next_state = self.model(current_state, pit) # given this state (board, turn, and history), what will the state (board, turn, and history) be if the the current player chooses this pit?
                
                next_player = PLAYER_ONE if next_state[player_index] < 0.5 else PLAYER_TWO # in that next state, who's turn is it?
                
                future_values_for_this_choice = self.get_future_values(next_state, num_turns_in_advance-1) # this evaluatates how good each option is for the current player
                
                if self.is_training:
                    # if training use softmax to explore better

                    if next_player == self.player_number:
                        # when I go again, max of options
                        probs = self.softmax(future_values_for_this_choice)

                    else:
                        # when it will be my opponents turn, min options
                        inverted_values = [-v for v in future_values_for_this_choice]
                        probs = self.softmax(inverted_values)

                    selected_index = np.random.choice(len(future_values_for_this_choice), p=probs)
                
                else:
                    # not training
                    if next_player == self.player_number:
                        selected_index = future_values_for_this_choice.index(max(future_values_for_this_choice))
                    
                    else:
                        # opponents turn
                        selected_index = future_values_for_this_choice.index(min(future_values_for_this_choice))

                if current_player == PLAYER_TWO:
                    pit = (pit - len(PLAYER_TWO_PITS) - 1)

                action_values[pit] = future_values_for_this_choice[selected_index]
           
            return action_values
        
        else:

            for pit in current_players_pits:
                policy_logits, value, next_state = self.model(current_state, pit) # choosing this pit, what will the board, who's turn, and history be?

                next_player = PLAYER_ONE if next_state[player_index] < 0.5 else PLAYER_TWO # in that next state, who's turn is it?

                policy_logits, value, final_state = self.model(next_state) # with that board, turn, and history, how good is that state for the current player?

                if current_player == PLAYER_TWO:
                    pit = (pit - len(PLAYER_TWO_PITS) - 1)

                if next_player == self.player_number:
                    # if it will be me, max choice
                    action_values[pit] = value.item()
                
                else:
                    # if it is the opponent, big is bad
                    action_values[pit] = -value.item()

            return action_values # return how good that board, turn, and state would be for each pit you can chose.
        

    def softmax(self, x: list[float]) -> list[float]:
        x = np.array(x)
        e_x = np.exp(x - np.max(x))  # for numerical stability
        return (e_x / e_x.sum()).tolist()
        

    def choose_move(self, board: list[int], gui: "MancalaGUI") -> None:
        '''
        This function is called from the gui to request a move from the model.
        '''

        # explore all choices FUTURE_TURNS_TO_PREDICT to see which will lead to the most likely win
        value_logits: list[float] = torch.tensor(self.get_future_values(gui.game.get_current_game_state(), FUTURE_TURNS_TO_PREDICT), dtype=torch.float32).clone()

        # exclude illegal moves
        value_logits[~self.generate_illegal_moves_mask(board)] = float('-inf')
        chosen_pit = torch.argmax(value_logits).item()

        if self.player_number == PLAYER_TWO:
            # adjust to other side of board
            chosen_pit = (chosen_pit + len(PLAYER_ONE_PITS) + 1)

        gui.make_move(chosen_pit)

    def export_model(self):
        # save model locally
        local_model_path = os.path.join(self.model.models_directory, f'{EXPORT_FILENAME}{self.model.number_of_games}.pth')
        local_loss_path = os.path.join(self.model.models_directory, f'{EXPORT_LOSSES_FILENAME}{self.model.number_of_games}.pkl')

        torch.save({
            'model_state_dict': self.model.state_dict(), 
            'optimizer_state_dict': self.model.optimizer.state_dict(),
        }, local_model_path)

        with open(local_loss_path, 'wb') as f:
            pickle.dump({
                'policy_losses': self.model.policy_losses,
                'value_losses': self.model.value_losses,
                'transition_losses': self.model.transition_losses,
                'number_of_games': self.model.number_of_games
            }, f)

        try:
            in_colab = 'google.colab' in str(get_ipython())
        except:
            in_colab = False

        # check if in Colab and Drive is mounted
        if in_colab:
            drive_model_dir = '/content/drive/MyDrive/MancalaModels'
            os.makedirs(drive_model_dir, exist_ok=True)

            drive_model_path = os.path.join(drive_model_dir, f'{EXPORT_FILENAME}{self.model.number_of_games}.pth')
            drive_loss_path = os.path.join(drive_model_dir, f'{EXPORT_LOSSES_FILENAME}{self.model.number_of_games}.pkl')

            # copy to Drive
            torch.save({
                'model_state_dict': self.model.state_dict(), 
                'optimizer_state_dict': self.model.optimizer.state_dict(),
            }, drive_model_path)

            with open(drive_loss_path, 'wb') as f:
                pickle.dump({
                    'policy_losses': self.model.policy_losses,
                    'value_losses': self.model.value_losses,
                    'transition_losses': self.model.transition_losses,
                    'number_of_games': self.model.number_of_games
                }, f)



class AlphaMancalaMind(nn.Module):
    '''
    This class is the neural network 
    behind the mancala agent.
    '''
    _instance = None

    def __new__(cls, *args, **kwargs):
        '''
        This dunder method enforces a Singleton 
        pattern for the neural network class.
        '''

        if cls._instance is None:
            cls._instance = super(AlphaMancalaMind, cls).__new__(cls)
        return cls._instance
       

    def __init__(self, hidden_size=128, lr=0.001, model_number_to_use=None) -> None:
        '''
        Constructor for the neural network.
        Initializes the architecture.
        '''
        # singleton, if an instance already exists, don't create another
        if getattr(self, '_initialized', False):
            return
        
        # init as an nn.Module
        super().__init__()

        # directory where models are stored
        self.models_directory = os.path.join(os.path.dirname(os.path.abspath(__file__)), EXPORT_FOLDER)
        os.makedirs(self.models_directory, exist_ok=True)


        # setup network
        board_size: int = len(PLAYER_ONE_PITS) + len(PLAYER_TWO_PITS) + 2

        # board size (len(PLAYER_ONE_PITS) + len(PLAYER_TWO_PITS) + 2), player number (1), history
        input_size: int = board_size + 1 + HISTORY_LENGTH

        super(AlphaMancalaMind, self).__init__()

        # network that encodes board information for all heads
        self.encoder = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU()
        )

        # Network that outputs logits across the 6 possible pit options
        self.policy_head = nn.Sequential(
            nn.Linear(hidden_size, len(PLAYER_ONE_PITS)),
        )

        # Network predicting how good this current game state is for us (-1, 1)
        self.value_head = nn.Sequential(
            nn.Linear(hidden_size, 1),
            nn.Tanh()
        )

        # network predicting the next board state after our move (the opponents next given state)
        self.transition_model = nn.Sequential(
            nn.Linear(hidden_size + len(PLAYER_ONE_PITS), hidden_size), # hidden_size + the one hot encoded vector of what action the policy_head chose
            nn.ReLU(),
            nn.Linear(hidden_size, board_size + 1 + HISTORY_LENGTH),  # same shape as game state
        )

        # adam optimizer
        self.optimizer = optim.Adam(self.parameters(), lr=lr)

        exported_models = os.listdir(self.models_directory)

        if len(exported_models) > 0:
            # if there is an exported model stored

            if model_number_to_use is not None:
                # if a model number was provided, use it
                if f'{EXPORT_FILENAME}{model_number_to_use}.pth' in os.listdir(self.models_directory):
                    checkpoint = torch.load(os.path.join(self.models_directory, f'{EXPORT_FILENAME}{model_number_to_use}.pth'))
                    self.load_state_dict(checkpoint['model_state_dict'])
                    self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

                    with open(f'{EXPORT_LOSSES_FILENAME}{model_number_to_use}.pkl', 'rb') as f:
                        data = pickle.load(f)
                        self.policy_losses = data['policy_losses']
                        self.value_losses = data['value_losses']
                        self.transition_losses = data['transition_losses']
                        self.number_of_games = data['number_of_games']
                        
                else:
                    print(f"{EXPORT_FILENAME}{model_number_to_use}.pth doesn't exist")
                    sys.exit()

            else:
                # if no number is given, assume the latest model

                # find the name of the highest model
                numbers = []
                for model in exported_models:
                    match = re.search(fr'{EXPORT_FILENAME}(\d+)\.pth', model)
                    if match:
                        numbers.append((int(match.group(1)), model))


                # load in model
                checkpoint = torch.load(os.path.join(self.models_directory, max(numbers)[1]))
                self.load_state_dict(checkpoint['model_state_dict'])
                self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

                with open(os.path.join(self.models_directory, f'{EXPORT_LOSSES_FILENAME}{max(numbers)[0]}.pkl'), 'rb') as f:
                    data = pickle.load(f)
                    self.policy_losses = data['policy_losses']
                    self.value_losses = data['value_losses']
                    self.transition_losses = data['transition_losses']
                    self.number_of_games = data['number_of_games']


        else:
            # if no model is stored, init all to 0/empty
            self.number_of_games = 0
            self.policy_losses = []
            self.value_losses = []
            self.transition_losses = []


        self._initialized = True


    def forward(self, board_state, action: int = None):
        '''
        Given a board state and the action taken, output: 
        probability distribution across all potential pits, 
        how good this current board state is for us, 
        reward gained after taking this action, state 
        after taking this action.
        '''

        # pass through the encoder to process board state generally
        x = self.encoder(board_state)


        policy_logits = self.policy_head(x) # given this state, which move should we take?
        value = self.value_head(x) # how good is the given board state for us?

        if action is not None:
            if action in PLAYER_TWO_PITS:
                # convert action pit back to range expected by model
                action = action - len(PLAYER_ONE_PITS) - 1
            action = F.one_hot(torch.tensor(action), num_classes=len(PLAYER_ONE_PITS)).float()
            
            # because I am not using batches, I need to add another dimention
            x = x.unsqueeze(0)
            action = action.unsqueeze(0)

            next_state = self.transition_model(torch.cat([x, action], dim=1)).squeeze(0) # what state will our opponent be given?
        
        else:
            # if just predicting what action to take
            next_state = None


        return policy_logits, value, next_state
