
from tkinter import *
from tkinter import ttk

from matplotlib.pyplot import grid

root = Tk()
frm = ttk.Frame(root, padding=10)
frm.grid()
ttk.Label(frm, text="1D Solver").grid(column=0, row=0)
ttk.Button(frm, text="Quit", command=root.destroy).grid(column=0, row=2)
ttk.Button(frm,text = "Solve", command=(print("Solving"))).grid(column= 0,row = 1)
root.mainloop()