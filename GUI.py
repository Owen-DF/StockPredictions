
from tkinter import *




root = Tk()  


root.title("Stock Predictions")
root.geometry("800x600")


title = Label(root, text = "Stock Predictions", font = ("Helvetica", 24), )
title.pack()


entry = Entry(root, width = 50)
entry.pack()


def clicked():
    title.configure(text = "Searching for " + entry.get() +" ...")
    



search = Button(root, text = "search", command = clicked, fg = "blue")
search.pack()
root.mainloop()
