from tkinter import *

import os
from tkinter.font import BOLD

import cv2
import time
import numpy as np


import tensorflow as tf
from tensorflow.keras import datasets, layers, models,preprocessing





window=Tk()

#-------background---------#
bg = PhotoImage(file="sign2.png")
label1 = Label( window, image = bg)
label1.place(x = -2, y = 0)


#-------navigation--------#
def main_nav(master):
    master.destroy()
    os.system("python main.py")


#----------Predict Function--------#
def proceed(master,input1):
    a=input1
    print(a)
    if(a=='N'):
        classifier = models.load_model('Numbers.h5')
        labels=['0','1','2','3','4','5','6','7','8','9']
    if(a=='A'):
        classifier = models.load_model('Alphabet.h5')
        labels=['A','B','C','D','E','F','G','H','I','K','L','O','P','Q','R','U','V','W','X','Y']
    if(a=='P'):
        classifier = models.load_model('Phrase.h5')
        labels=['Call me','Dislike','Good Job','Good Luck','Loser','Peace','Rock','Shocker']

    count=0

    #classifier = models.load_model('mymodel.h5')

    image_x, image_y = 64, 64

    def nothing(x):
        pass

    cam = cv2.VideoCapture(0,cv2.CAP_DSHOW)
    #cv2.namedWindow("test")

    cv2.namedWindow("Trackbars")
    cv2.resizeWindow("Trackbars",600,350)
    cv2.createTrackbar("L - H", "Trackbars", 0, 179, nothing)
    cv2.createTrackbar("L - S", "Trackbars", 0, 255, nothing)
    cv2.createTrackbar("L - V", "Trackbars", 0, 255, nothing)
    cv2.createTrackbar("U - H", "Trackbars", 179, 179, nothing)
    cv2.createTrackbar("U - S", "Trackbars", 255, 255, nothing)
    cv2.createTrackbar("U - V", "Trackbars", 255, 255, nothing)
    
    while True:
            ret, frame = cam.read()
            frame = cv2.flip(frame, 1)

            l_h = cv2.getTrackbarPos("L - H", "Trackbars")
            l_s = cv2.getTrackbarPos("L - S", "Trackbars")
            l_v = cv2.getTrackbarPos("L - V", "Trackbars")
            u_h = cv2.getTrackbarPos("U - H", "Trackbars")
            u_s = cv2.getTrackbarPos("U - S", "Trackbars")
            u_v = cv2.getTrackbarPos("U - V", "Trackbars")

            img = cv2.rectangle(frame, (425, 100), (625, 300), (0, 255, 0), thickness=2, lineType=8, shift=0)
            lower_blue = np.array([l_h, l_s, l_v])
            upper_blue = np.array([u_h, u_s, u_v])
            imcrop = img[102:298, 427:623]
            hsv = cv2.cvtColor(imcrop, cv2.COLOR_BGR2HSV)
            mask = cv2.inRange(hsv, lower_blue, upper_blue)
            #cv2.putText(frame, str(img_counter), (30, 400), cv2.FONT_HERSHEY_TRIPLEX, 1.5, (127, 127, 255))
            result = cv2.bitwise_and(imcrop, imcrop, mask=mask)

            count=0
            #cv2.imshow("test", frame)
            cv2.imshow("mask", mask)
            #cv2.imshow("result", result)
            if cv2.waitKey(1) == ord('c'):
                img_name = "{}.png".format(count)
                save_img = cv2.resize(mask, (image_x, image_y))
                cv2.imwrite(img_name, save_img)
                pat="0.png"
                test_image = preprocessing.image.load_img(pat, target_size=(64, 64))
                test_image = preprocessing.image.img_to_array(test_image)
                test_image = np.expand_dims(test_image, axis = 0)
                result = classifier.predict(test_image)
                result=result.tolist()
                v=max(result[0])
                ind=result[0].index(v)
                print(labels[ind])
                Output.delete('1.0', END)
                Output.insert(END, labels[ind])
                #cv2.destroyWindow("test")
                cv2.destroyWindow("mask")
                #cv2.destroyWindow("result")
                cv2.destroyWindow("Trackbars")
                cam.release()
                break


#---------Interface-----------#

lbl=Label(window, text="Sign Language Predictor",fg='white',bg="Black", font=("Helvetica", 24))
lbl.place(x=300, y=20)

lbl1=Label(window, text="Sign Type",fg='white',bg="Black", font=("Helvetica", 15))
lbl1.place(x=350, y=90)


txtfld=Entry(window, text="This is Entry Widget",bd=5,border=0,font=("Helvetica", 15))
txtfld.place(x=500, y=90)

btn=Button(window, text="Predict", fg='white',bg='#005BBB',command=lambda:proceed(window,txtfld.get()),height=1,width=10,font=("Helvetica", 15,BOLD))
btn.place(x=420, y=140)


Output = Text(window, height = 1.1, width = 15, fg='blue',font=("Helvetica", 15,BOLD))
Output.place(x=400, y=195)


btn2=Button(window, text="Exit", fg='white',bg='red',command=lambda:main_nav(window),height=1,width=8,font=("Helvetica", 15,BOLD))
btn2.place(x=430, y=250)



window.resizable(width="False",height="False")
window.title('Sign Predictor')
window.geometry("900x500+400+20")


window.mainloop()