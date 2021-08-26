from tkinter import *
from tkinter import messagebox
import os
from tkinter.font import BOLD

window=Tk()
#-------background---------#
bg = PhotoImage(file="sign2.png")
label1 = Label( window, image = bg)
label1.place(x = -2, y = 0)

#-------navigation--------#
def pred(master):
    master.destroy()
    os.system("python predict.py")

def cap(master):
    master.destroy()
    os.system("python capture.py")

def train(master):
    master.destroy()
    os.system("python train.py")


#---------Interface-----------#
lbl=Label(window, text="Sign Language Predictor", fg='white',bg="Black", font=("Helvetica", 24))
lbl.place(x=300, y=20)

btn=Button(window, text="Predict", fg='white',bg='#005BBB',command=lambda:pred(window),height=2,width=10,font=("Helvetica", 15,BOLD))
btn.place(x=150, y=130)

btn1=Button(window, text="Capture", fg='white',bg='#005BBB',command=lambda:cap(window),height=2,width=10,font=("Helvetica", 15,BOLD))
btn1.place(x=400, y=130)

btn2=Button(window, text="Train", fg='white',bg='#005BBB',command=lambda:train(window),height=2,width=10,font=("Helvetica", 15,BOLD))
btn2.place(x=650, y=130)

window.resizable(width="False",height="False")
window.title('Sign Predictor')
window.geometry("900x500+400+20")


window.mainloop()