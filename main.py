from tkinter import *
import ttkbootstrap as ttk
import grumodel



#functions

def entryClicked():
    company = clicked.get()
    grumodel.readFile(company)
    return










#set up
root = ttk.Window(themename = "superhero")
root.title("EasyInvest")
root.geometry('600x600')


#Title
title = ttk.Label(root, text="EasyInvest")
title.pack(pady=30)
title.config(font=("Times New Roman", 20, "bold"))



#Dropdown
dropdownFrame = ttk.Frame(root)
dropdownFrame.pack(padx=5, pady=10, fill='x')
ttk.Label(dropdownFrame,  text = "Select Company:").pack(side="left", padx = 5)
#these are the current companies we will track and train the models for
options = ['Nvidia', 'IBM', 'Whirlpool', 'United Rentals']
clicked = StringVar()
clicked.set("Select a company")
dropdown = OptionMenu(dropdownFrame, clicked, *options)
dropdown.pack(side = "left")

entryButton = Button(dropdownFrame, text = "Submit", command = entryClicked)
entryButton.pack(side = "left", padx = 5)

root.mainloop()