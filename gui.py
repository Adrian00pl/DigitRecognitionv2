import os
from tkinter import *
import PIL
from PIL import ImageGrab,ImageDraw
import przetwarzanie


class main:
    def __init__(self, master):
        self.master = master
        self.res = ""
        self.pre = [None, None]
        self.bs = 9
        self.c = Canvas(self.master,bd=3,relief="ridge", width=560, height=560, bg='Gainsboro')
        self.rectangle()
        self.c.pack(side=LEFT)
        f1 = Frame(self.master, padx=20, pady=20)
        self.pr = Label(f1,text="Prediction:",fg="black",font=("",20,"bold"))
        self.pr.pack(pady=20)



        Button(f1,font=("",10),fg="white",bg="black", text="Wyczysc", command=self.clear_all).pack(side=BOTTOM)
        Button(f1,font=("",10),fg="white",bg="black", text="Rozpoznaj", command=self.getResult).pack(side=BOTTOM)
        f1.pack(side=RIGHT,fill=Y)
        self.c.bind("<B1-Motion>", self.draw_lines)



    def getResult(self):
        box = (self.c.winfo_rootx(), self.c.winfo_rooty(), self.c.winfo_rootx() + self.c.winfo_width(),
               self.c.winfo_rooty() + self.c.winfo_height())
        grab = ImageGrab.grab(bbox=box)
        grab.save('znak.png')
        self.res = str(przetwarzanie.predict("znak.png"))
        self.pr['text'] = "Prediction: " + self.res



    def rectangle(self):
        for y in range(2,560,20):
            for x in range(2,560,20):
                self.c.create_rectangle(x,y,x+20,y+20,fill="White", width=1,outline="Gainsboro")

    def draw_lines(self, event):
        self.x = event.x
        self.y = event.y
        r = 12
        self.c.create_oval(self.x - r, self.y - r, self.x + r, self.y + r, fill='black')

    def clear_all(self):
        self.c.delete("all")
        self.rectangle()


if __name__ == "__main__":
    root = Tk()
    main(root)
    root.title('Rozpoznawanie liczb')
    root.resizable(0, 0)
    root.mainloop()
