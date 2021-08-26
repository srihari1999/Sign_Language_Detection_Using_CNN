from tkinter import *
from tkinter import messagebox
import os
from tkinter.font import BOLD

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

#------------Model Training------------#
def alpha(master):
    model = models.Sequential()
    model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape = (64, 64,3)))
    model.add(layers.MaxPooling2D((2, 2)))

    model.add(layers.Conv2D(32, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))

    model.add(layers.Conv2D(64, (3, 3), activation='relu',padding="same"))
    model.add(layers.MaxPooling2D((2, 2),padding="same"))

    model.add(layers.Flatten())
    model.add(layers.Dense(256, activation='relu'))
    model.add(layers.Dropout(0.5))
    model.add(layers.Dense(20, activation='softmax'))


    model.compile(optimizer='sgd',
                loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True),
                metrics=['accuracy'])

    train_datagen = preprocessing.image.ImageDataGenerator(
            rescale=1./255,
            shear_range=0.2,
            zoom_range=0.2,
            horizontal_flip=True)
    test_datagen = preprocessing.image.ImageDataGenerator(rescale=1./255)
    train_generator = train_datagen.flow_from_directory(
            'mydata/Alphabets/training_set',
            target_size=(64, 64),
            batch_size=32,
            class_mode='categorical')
    validation_generator = test_datagen.flow_from_directory(
            'mydata/Alphabets/test_set',
            target_size=(64, 64),
            batch_size=32,
            class_mode='categorical')
    model.fit(
            train_generator,
            steps_per_epoch=407,
            epochs=10,
            validation_data=validation_generator,
            validation_steps=219)


    model.save('Alphabet.h5')
    lbl=Label(window, text="Completed", fg='red',bg='black', font=("Helvetica", 16))
    lbl.place(x=400, y=100)
    #master.destroy()
    #os.system("python predict.py")

def num(master):
    model = models.Sequential()
    model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape = (64, 64,3)))
    model.add(layers.MaxPooling2D((2, 2)))

    model.add(layers.Conv2D(32, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))

    model.add(layers.Conv2D(64, (3, 3), activation='relu',padding="same"))
    model.add(layers.MaxPooling2D((2, 2),padding="same"))

    model.add(layers.Flatten())
    model.add(layers.Dense(256, activation='relu'))
    model.add(layers.Dropout(0.5))
    model.add(layers.Dense(10, activation='softmax'))

    model.compile(optimizer='sgd',
                loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True),
                metrics=['accuracy'])

    train_datagen = preprocessing.image.ImageDataGenerator(
            rescale=1./255,
            shear_range=0.2,
            zoom_range=0.2,
            horizontal_flip=True)
    test_datagen = preprocessing.image.ImageDataGenerator(rescale=1./255)
    train_generator = train_datagen.flow_from_directory(
            'mydata/Numbers/training_set',
            target_size=(64, 64),
            batch_size=32,
            class_mode='categorical')
    validation_generator = test_datagen.flow_from_directory(
            'mydata/Numbers/test_set',
            target_size=(64, 64),
            batch_size=32,
            class_mode='categorical')
    model.fit(
            train_generator,
            steps_per_epoch=204,
            epochs=10,
            validation_data=validation_generator,
            validation_steps=110)

    model.save('Numbers.h5')
    lbl=Label(window, text="Completed", fg='red',bg='black', font=("Helvetica", 16))
    lbl.place(x=400, y=100)

def phrase(master):
        model = models.Sequential()
        model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape = (64, 64,3)))
        model.add(layers.MaxPooling2D((2, 2)))

        model.add(layers.Conv2D(32, (3, 3), activation='relu'))
        model.add(layers.MaxPooling2D((2, 2)))

        model.add(layers.Conv2D(64, (3, 3), activation='relu',padding="same"))
        model.add(layers.MaxPooling2D((2, 2),padding="same"))

        model.add(layers.Flatten())
        model.add(layers.Dense(256, activation='relu'))
        model.add(layers.Dropout(0.5))
        model.add(layers.Dense(8, activation='softmax'))


        model.compile(optimizer='sgd',
                loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True),
                metrics=['accuracy'])

        train_datagen = preprocessing.image.ImageDataGenerator(
                rescale=1./255,
                shear_range=0.2,
                zoom_range=0.2,
                horizontal_flip=True)
        test_datagen = preprocessing.image.ImageDataGenerator(rescale=1./255)
        train_generator = train_datagen.flow_from_directory(
                'mydata/Phrases/training_set',
                target_size=(64, 64),
                batch_size=32,
                class_mode='categorical')
        validation_generator = test_datagen.flow_from_directory(
                'mydata/Phrases/test_set',
                target_size=(64, 64),
                batch_size=32,
                class_mode='categorical')
        model.fit(
                train_generator,
                steps_per_epoch=163,
                epochs=10,
                validation_data=validation_generator,
                validation_steps=88)


        model.save('Phrase.h5')
        lbl=Label(window, text="Completed", fg='red',bg='black', font=("Helvetica", 16))
        lbl.place(x=400, y=100)
#---------Interface-----------#

lbl=Label(window, text="Sign Language Predictor", fg='white',bg="Black", font=("Helvetica", 24))
lbl.place(x=300, y=20)

btn=Button(window, text="Alphabets", fg='white',bg='#005BBB',command=lambda:alpha(window),height=2,width=10,font=("Helvetica", 15,BOLD))
btn.place(x=200, y=150)

btn1=Button(window, text="Numbers", fg='white',bg='#005BBB',command=lambda:num(window),height=2,width=10,font=("Helvetica", 15,BOLD))
btn1.place(x=400, y=150)

btn1=Button(window, text="Phrases", fg='white',bg='#005BBB',command=lambda:phrase(window),height=2,width=10,font=("Helvetica", 15,BOLD))
btn1.place(x=600, y=150)

btn2=Button(window, text="Exit", fg='white',bg='#005BBB',command=lambda:main_nav(window),height=1,width=10,font=("Helvetica", 15,BOLD))
btn2.place(x=400, y=250)

window.resizable(width="False",height="False")
window.title('Sign Predictor')
window.geometry("900x500+400+20")

window.mainloop()