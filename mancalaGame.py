from collections import deque
import torch
from gameEnvironment import PLAYER_ONE, PLAYER_TWO, PLAYER_ONE_STORE, PLAYER_TWO_STORE, PLAYER_ONE_PITS, PLAYER_TWO_PITS, HISTORY_LENGTH


class MancalaGame:
    def __init__(self):
        '''
        initialize a mancala game to be played.
        '''

        # setting up game
        self.board = [4] * len(PLAYER_ONE_PITS) + [0] + [4] * len(PLAYER_TWO_PITS) + [0]
        self.current_player = PLAYER_ONE
        self.winner = None
        self.move_history: deque[int] = deque([-1] * HISTORY_LENGTH, maxlen=HISTORY_LENGTH)
        self.whole_history: list[MancalaTurn] = []

    def get_current_game_state(self) -> list[int]:
        return torch.tensor(self.board + [self.current_player] + list(self.move_history), dtype=torch.float32) 

    def move(self, pit: int) -> str | bool:
        '''
        this function checks that the current player chose a valid pit
        and stores all meta information about the move for future training.
        '''

        # gather information about game before turn
        player_moving = self.current_player
        state_before: list[int] = self.get_current_game_state()
        score_before = self.board[PLAYER_ONE_STORE] if player_moving == PLAYER_ONE else self.board[PLAYER_TWO_STORE]


        # make move
        if ((player_moving == PLAYER_ONE) and (pit in PLAYER_ONE_PITS)) or ((player_moving == PLAYER_TWO) and (pit in PLAYER_TWO_PITS)):
            # check that it is a valid move
            ret_val = self.process_move(pit)
        else:
            ret_val = "Invalid move. Pick a pit from your side."

        
        # gather information about game after turn
        state_after: list[int] = self.get_current_game_state()
        score_after = self.board[PLAYER_ONE_STORE] if player_moving == PLAYER_ONE else self.board[PLAYER_TWO_STORE]


        # record information
        self.move_history.append(pit)
        self.whole_history.append(MancalaTurn(
            player_moving,
            state_before,
            pit,
            state_after,
            score_after-score_before
        ))

        return ret_val


    def process_move(self, pit: int) -> str | bool:
        '''
        This function makes a move for a player.
        '''

        if self.board[pit] == 0:
            return "You cannot pick an empty pit. Try again."
        
        stones = self.board[pit]
        self.board[pit] = 0
        index = pit
        
        while stones > 0:
            index = (index + 1) % len(self.board)
            if (self.current_player == PLAYER_ONE and index == PLAYER_TWO_STORE) or (self.current_player == PLAYER_TWO and index == PLAYER_ONE_STORE):
                continue  # Skip opponent's store
            self.board[index] += 1
            stones -= 1
        
        # Check for capture move
        self.check_capture(index)
        
        # Check if the game should end
        if self.check_game_end():
            return True
        
        # Switch turns unless last stone landed in own store
        if not (self.current_player == PLAYER_ONE and index == PLAYER_ONE_STORE or self.current_player == PLAYER_TWO and index == PLAYER_TWO_STORE):
            self.current_player = 1 - self.current_player
        return True
    
    def check_capture(self, index: int) -> None:
        '''
        After a players turn, this function checks
        if a capture should occur.
        '''

        if self.current_player == PLAYER_ONE and index in PLAYER_ONE_PITS and self.board[index] == 1:
            opposite_index = (len(self.board)-2) - index
            if self.board[opposite_index] > 0:
                self.board[PLAYER_ONE_STORE] += self.board[opposite_index] + 1
                self.board[opposite_index] = 0
                self.board[index] = 0
        elif self.current_player == PLAYER_TWO and index in PLAYER_TWO_PITS and self.board[index] == 1:
            opposite_index = (len(self.board)-2) - index
            if self.board[opposite_index] > 0:
                self.board[PLAYER_TWO_STORE] += self.board[opposite_index] + 1
                self.board[opposite_index] = 0
                self.board[index] = 0
    
    def check_game_end(self) -> bool:
        '''
        After each turn, this function checks if the game is over.
        '''
        
        if all(self.board[i] == 0 for i in PLAYER_ONE_PITS) or all(self.board[i] == 0 for i in PLAYER_TWO_PITS):
            # Collect remaining stones
            self.board[PLAYER_ONE_STORE] += sum(self.board[i] for i in PLAYER_ONE_PITS)
            self.board[PLAYER_TWO_STORE] += sum(self.board[i] for i in PLAYER_TWO_PITS)
            for i in PLAYER_ONE_PITS:
                self.board[i] = 0
            for i in PLAYER_TWO_PITS:
                self.board[i] = 0

            if self.board[PLAYER_ONE_STORE] > self.board[PLAYER_TWO_STORE]:
                self.winner = PLAYER_ONE
            elif  self.board[PLAYER_ONE_STORE] < self.board[PLAYER_TWO_STORE]:
                self.winner = PLAYER_TWO
            else:
                self.winner = 3
            return True
        return False



class MancalaTurn():
    '''
    This is a class grouping information about a turn together.
    '''
    def __init__(self, player, state_before, action, state_after, reward):
        self.player = player
        self.state_before = state_before
        self.action = action
        self.state_after = state_after
        self.reward = reward
