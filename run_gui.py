import tkinter as tk

from src.calibgui import CalibGUI

if __name__=='__main__':

    root = tk.Tk()
    app = CalibGUI(root)
    root.title('Deeping Source Calibration GUI')
    root.resizable(False, False)

    root.mainloop()
