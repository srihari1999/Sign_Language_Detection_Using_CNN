from tkinter import *
from tkinter.font import BOLD
from cv2 import cv2
import time
import numpy as np
import os

from tkinter import messagebox

window=Tk()

#-------background---------#
bg = PhotoImage(file="sign2.png")
label1 = Label( window, image = bg)
label1.place(x = -2, y = 0)

#-------navigation--------#
def main_nav(master):
    master.destroy()
    os.system("python main.py")

def proceed(master,inp,ges):
    capture_images(ges,inp)


#----------Capture Functions---------#

image_x, image_y = 64, 64

def nothing(x):
    pass


def create_folder(folder_name,input_type):
    if(input_type=='N'):
        if not os.path.exists('./mydata/Numbers/training_set/' + folder_name):
            os.mkdir('./mydata/Numbers/training_set/' + folder_name)
        if not os.path.exists('./mydata/Numbers/test_set/' + folder_name):
            os.mkdir('./mydata/Numbers/test_set/' + folder_name)
    if(input_type=='A'):
        if not os.path.exists('./mydata/Alphabets/training_set/' + folder_name):
                os.mkdir('./mydata/Alphabets/training_set/' + folder_name)
        if not os.path.exists('./mydata/Alphabets/test_set/' + folder_name):
            os.mkdir('./mydata/Alphabets/test_set/' + folder_name)
    if(input_type=='P'):
        if not os.path.exists('./mydata/Phrases/training_set/' + folder_name):
                os.mkdir('./mydata/Phrases/training_set/' + folder_name)
        if not os.path.exists('./mydata/Phrases/test_set/' + folder_name):
            os.mkdir('./mydata/Phrases/test_set/' + folder_name)


def capture_images(ges_name,input_type):
    create_folder(str(ges_name),str(input_type))

    if input_type=='N':
        pathname='./mydata/Numbers/'
    elif input_type=='A':
        pathname='./mydata/Alphabets/'
    else:
        pathname='./mydata/Phrases/'

    img_counter=0
    t_counter=0
    listImage = [1,2,3,4,5]
    training_set_image_name = 1
    test_set_image_name = 1

    cam = cv2.VideoCapture(0,cv2.CAP_DSHOW)
    #cv2.namedWindow("test")

    cv2.namedWindow("Trackbars")
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
        #lower_blue = np.array([0, 0, 76])
        #upper_blue = np.array([179, 255, 255])
        lower_blue = np.array([l_h, l_s, l_v])
        upper_blue = np.array([u_h, u_s, u_v])
        imcrop = img[102:298, 427:623]
        hsv = cv2.cvtColor(imcrop, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, lower_blue, upper_blue)
        cv2.putText(frame, str(img_counter), (30, 400), cv2.FONT_HERSHEY_TRIPLEX, 1.5, (127, 127, 255))
        result = cv2.bitwise_and(imcrop, imcrop, mask=mask)

        
        #cv2.imshow("test", frame)
        cv2.imshow("mask", mask)
        #cv2.imshow("result", result)

        #for i in range(1001):
        if cv2.waitKey(1) == ord('c'):
            if t_counter <= 650:
                img_name = pathname+"training_set/" + str(ges_name) + "/{}.png".format(training_set_image_name)
                save_img = cv2.resize(mask, (image_x, image_y))
                cv2.imwrite(img_name, save_img)
                print("{} written!".format(img_name))
                training_set_image_name += 1


            if t_counter > 650 and t_counter <= 1000:
                img_name = pathname+"test_set/" + str(ges_name) + "/{}.png".format(test_set_image_name)
                save_img = cv2.resize(mask, (image_x, image_y))
                cv2.imwrite(img_name, save_img)
                print("{} written!".format(img_name))
                test_set_image_name += 1

            t_counter+=1
            img_counter+=1
        if(t_counter>=1001):
            messagebox.showinfo("Info","Done")
            cv2.destroyWindow("mask")
            cv2.destroyWindow("Trackbars")
            cam.release()
            break
            #exit("Successful")


#---------Interface-----------#

lbl=Label(window, text="Sign Language Predictor", fg='white',bg="Black", font=("Helvetica", 24))
lbl.place(x=300, y=20)

lb2=Label(window, text="Sign Type", fg='white',bg="Black", font=("Helvetica", 14))
lb2.place(x=300, y=100)

inp=Entry(window, text="Input type", bd=5,border=0,font=("Helvetica", 14))
inp.place(x=500, y=100)

lb3=Label(window, text="Sign Name", fg='white',bg="Black", font=("Helvetica", 14))
lb3.place(x=300, y=150)

ges=Entry(window, text="Gesture Name", bd=5,border=0,font=("Helvetica", 14))
ges.place(x=500, y=150)

btn=Button(window, text="Capture", fg='blue',command=lambda:proceed(window,inp.get(),ges.get()),height=1,width=10,font=("Helvetica", 15,BOLD))
btn.place(x=300, y=200)

btn1=Button(window, text="Exit", fg='white',bg="Red",command=lambda:main_nav(window),height=1,width=10,font=("Helvetica", 15,BOLD))
btn1.place(x=500, y=200)




window.resizable(width="False",height="False")
window.title('Sign Predictor')
window.geometry("900x500+400+20")

window.mainloop()