from mancalaGUI import MancalaGUI


def main(train):
    if train:
        MancalaGUI(train=train)
    else:
        import tkinter as tk
        root = tk.Tk()
        MancalaGUI(root=root, train=train)
        root.mainloop()


if __name__ == "__main__":
    train = True
    main(train)
