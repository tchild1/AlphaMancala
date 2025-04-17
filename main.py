from mancalaGUI import MancalaGUI
import tkinter as tk


def main(train):
    if train:
        root = None
        MancalaGUI(root, train)
    else:
        root = tk.Tk()
        MancalaGUI(root, train)
        root.mainloop()


if __name__ == "__main__":
    train = True
    main(train)
