# -*- coding: utf-8 -*-
# @Author: Ananth
# @Date:   2017-08-01 14:00:04
# @Last Modified by:   Ananth
# @Last Modified time: 2017-08-01 14:18:08
import tkinter as tk


class Application(tk.Frame):

    def __init__(self, master=None):
        tk.Frame.__init__(self, master)
        self.grid()
        self.createWidgets()

    def createWidgets(self):
        self.quitButton = tk.Button(self, text='Quit',
                                    command=self.quit)
        self.quitButton.grid()

app = Application()
app.master.title('Sample application')
app.mainloop()
