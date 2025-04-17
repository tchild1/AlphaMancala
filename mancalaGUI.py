import tkinter as tk
from tkinter import messagebox
from alphaMancala import AlphaMancala, AlphaMancalaMind
from mancalaGame import MancalaGame
from gameEnvironment import PLAYER_ONE, PLAYER_TWO, PLAYER_ONE_PITS, PLAYER_TWO_PITS, WAIT_TIME, PLAYER_ONE_STORE, PLAYER_TWO_STORE, EXPORT_EVERY_GAMES, NUMBER_OF_TRAINING_GAMES
import torch.nn.functional as F
import torch
from tqdm import tqdm


class MancalaGUI:
    def __init__(self, root, train=False) -> None:
        '''
        This fuction initializes the UI (or not if no UI is to be shown)
        regardless, this class acts as an interface between the players
        and the game logic.
        '''

        self.root: tk.Tk = root
        self.game: MancalaGame = MancalaGame()
        self.is_training: bool = False

        if train:
            self.train()
        else:
            self.create_play_type_prompt()


    def take_turn(self) -> None:
        '''
        This function keeps notifying players 
        to play as long as there is no winner.
        '''
        if self.game.winner is not None:
            return

        self.notify_next_player()

        if not self.is_training:
            self.root.after(WAIT_TIME, self.take_turn)
        else:
            self.take_turn()

    def play_again_question(self) -> None:
        self.play_again_window = tk.Toplevel(self.root)
        self.play_again_window.title("Play Again?")
        self.play_again_window.geometry("300x200")
        
        tk.Label(self.play_again_window, text="Play Again?\nSelect:", font=("Arial", 14)).pack(pady=10)
        
        yes_btn = tk.Button(
            self.play_again_window, 
            text="Yes", 
            font=("Arial", 12), 
            command=self.play_again_true
        )
        yes_btn.pack(pady=5)
    
        no_btn = tk.Button(
            self.play_again_window, 
            text="No", 
            font=("Arial", 12), 
            command=self.play_again_false
        )
        no_btn.pack(pady=5)

    def play_again_true(self) -> None:
        '''
        This function is called when the user requests to play again.
        '''
        self.play_again_window.destroy()

        self.game = MancalaGame()
        self.create_play_type_prompt()

    def play_again_false(self) -> None:
        '''
        This function is called when the user does not request to play again.
        '''
        exit()

    def create_play_type_prompt(self) -> None:
        '''
        When the UI is shown, it will first
        prompt the user to chose the play mode and 
        configure the players for gameplay
        '''

        # Create a window for selecting the play type
        self.play_type_window: tk.Toplevel = tk.Toplevel(self.root)
        self.play_type_window.title("Select Play Type")
        self.play_type_window.geometry("300x300")
        
        tk.Label(self.play_type_window, text="Select Play Type:", font=("Arial", 14)).pack(pady=10)
        
        player_vs_player_btn: tk.Button = tk.Button(
            self.play_type_window, 
            text="Player 1 v Player 2", 
            font=("Arial", 12), 
            command=self.start_player_vs_player
        )
        player_vs_player_btn.pack(pady=5)
    
        player_vs_bot_btn: tk.Button = tk.Button(
            self.play_type_window, 
            text="Player 1 v Bot", 
            font=("Arial", 12), 
            command=self.start_player_vs_bot
        )
        player_vs_bot_btn.pack(pady=5)

        bot_vs_bot_btn: tk.Button = tk.Button(
            self.play_type_window, 
            text="Bot v Bot", 
            font=("Arial", 12), 
            command=self.start_bot_vs_bot
        )
        bot_vs_bot_btn.pack(pady=5)


    def start_player_vs_player(self) -> None:
        '''
        This function is called when the 
        player v player option is chosen.
        '''

        self.player_one = 'player'
        self.player_two = 'player'

        self.play_type_window.destroy() # Close the play type selection window

        self.create_ui()  # Show the game UI

        self.take_turn()


    def start_player_vs_bot(self) -> None:
        '''
        This function is called when the 
        player v bot option is chosen.
        '''

        self.player_one = 'player'
        self.player_two = AlphaMancala(PLAYER_TWO)

        self.play_type_window.destroy() # Close the play type selection window
        
        self.create_ui()  # Show the game UI

        self.take_turn()

    
    def start_bot_vs_bot(self) -> None:
        '''
        This function is called when the bot v bot
        option is chosen.
        '''

        self.player_one = AlphaMancala(PLAYER_ONE)
        self.player_two = AlphaMancala(PLAYER_TWO)

        self.play_type_window.destroy() # Close the play type selection window
        
        self.create_ui()  # Show the game UI

        self.take_turn()
   
    def train(self) -> None:
        '''
        This function is called to train the model.
        '''
        self.is_training = True

        self.player_one = AlphaMancala(PLAYER_ONE, self.is_training)
        self.player_two = AlphaMancala(PLAYER_TWO, self.is_training)

        for game_number in tqdm(range(NUMBER_OF_TRAINING_GAMES), desc="Training games"):

            # play a game
            self.game = MancalaGame()
            self.take_turn()
            self.player_one.model.number_of_games += 1

            for move in self.game.whole_history:
                
                # get model and convert player 2 pits to correct range
                if move.player == PLAYER_ONE:
                    model: AlphaMancalaMind = self.player_one.model
                    action: int = move.action
                elif move.player == PLAYER_TWO:
                    model: AlphaMancalaMind = self.player_two.model
                    action: int = (move.action - len(PLAYER_ONE_PITS) - 1)


                # get predictions from model now that we know the action taken
                predicted_policy_logits, predicted_value, predicted_next_state = model(move.state_before, move.action)


                # how wrong was the policy head at predicting what the best move was?
                # here we are teaching the policy head to mimic the value head in chosing the next action 
                policy_loss = F.cross_entropy(predicted_policy_logits, torch.tensor(action, dtype=torch.long))
                
                # How good was the model at predicting the 'goodness' of the current state?
                # states leading to a win should be a 1, states leading to a loss should be -1
                value_loss = F.mse_loss(predicted_value, torch.tensor([1.0], dtype=torch.float32) if self.game.winner == move.player else torch.tensor([-1.0], dtype=torch.float32)) # tie? # use a reward signal (difference in scores)

                # how close was the model to predicting the next state after our move?
                # a low loss here helps the value model make informed decisions about what action to take.
                transition_loss = F.mse_loss(predicted_next_state, move.state_after)


                self.player_one.model.policy_losses.append(policy_loss.item())
                self.player_one.model.value_losses.append(value_loss.item())
                self.player_one.model.transition_losses.append(transition_loss.item())


                # calculate total loss
                # policy loss: teaches which actions were taken in those states
                # value loss: teaches which states are good or bad
                # transition loss: teaches to predict the next state for an action
                total_loss = policy_loss + value_loss + transition_loss

                # optimize model
                model.optimizer.zero_grad()
                total_loss.backward()
                model.optimizer.step()


            if game_number % EXPORT_EVERY_GAMES == 0:
                self.player_one.export_model()

        self.player_one.export_model()
        self.is_training = False
        
        exit()


    def create_ui(self) -> None:
        '''
        This function creates the UI board
        '''
        self.root.title("Mancala")
        self.root.configure(bg="#DAA520")
        self.update_board()


    def update_board(self) -> None:
        '''
        After each move, this function updates the game board
        '''

        for widget in self.root.winfo_children():
            widget.destroy()
        
        # Stores
        p2_store: tk.Label = tk.Label(
            self.root, 
            text=str(self.game.board[13]), 
            font=("Arial", 14, "bold"), 
            bg="#B8860B", 
            fg="white", 
            width=5, 
            height=3
        )
        p2_store.grid(row=1, column=0, rowspan=2)
        
        p1_store: tk.Label = tk.Label(
            self.root, 
            text=str(self.game.board[6]), 
            font=("Arial", 14, "bold"), 
            bg="#B8860B", 
            fg="white", 
            width=5, 
            height=3
        )
        p1_store.grid(row=1, column=7, rowspan=2)
        
        # Player 2's pits (top row, right to left)
        for i in range(12, 6, -1):
            bg_color = "#32CD32" if self.game.current_player == 1 and self.game.board[i] > 0 else "SystemButtonFace"
            tk.Button(
                self.root, 
                text=str(self.game.board[i]), 
                font=("Arial", 12), 
                width=5, 
                height=2, 
                bg=bg_color, 
                command=lambda i=i: self.make_move(i)
            ).grid(row=1, column=13-i)
        
        # Player 1's pits (bottom row, left to right)
        for i in range(0, 6):
            bg_color = "#32CD32" if self.game.current_player == 0 and self.game.board[i] > 0 else "SystemButtonFace"
            tk.Button(
                self.root, 
                text=str(self.game.board[i]), 
                font=("Arial", 12), 
                width=5, 
                height=2, 
                bg=bg_color, 
                command=lambda i=i: self.make_move(i)
            ).grid(row=2, column=i+1)
        
        # Player turn display
        tk.Label(self.root, text=f"Current Player: {'Player 1' if self.game.current_player == 0 else 'Player 2'}", font=("Arial", 12, "bold"), bg="#DAA520").grid(row=3, column=1, columnspan=6)
    
    def make_move(self, pit: int) -> None:
        '''
        This function handles a move made by either player or bot
        '''

        result = self.game.move(pit)
        if result is not True:
            if not self.is_training:
                messagebox.showerror("Invalid Move", result)
            else:
                print('Invalid Move: ', result)
                print('Cannot handle wrong move')
                exit()

        if not self.is_training:
            self.update_board()

        if self.game.winner is not None:
            winner = f'Player {self.game.winner+1}'
            if not self.is_training:
                messagebox.showinfo("Game Over", f"Game over! {winner} wins!")
                self.play_again_question()
            else:
                if not self.is_training:
                    print(f"Game over! {winner} wins!")

    def notify_next_player(self) -> None:
        '''
        After each turn this function will notify the next player
        to take a turn.
        '''

        if self.game.current_player == 0 and isinstance(self.player_one, AlphaMancala):
            self.player_one.choose_move(self.game.board, self)
        
        elif self.game.current_player == 1 and isinstance(self.player_two, AlphaMancala):
            self.player_two.choose_move(self.game.board, self)
