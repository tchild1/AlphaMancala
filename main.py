from mancalaGUI import MancalaGUI


def main(train):
    if train:
        root = None
        MancalaGUI(root, train)
    else:
        import tkinter as tk
        root = tk.Tk()
        MancalaGUI(root, train)
        root.mainloop()


if __name__ == "__main__":
    train = True
    main(train)
