import tkinter
import cv2
import PIL.Image,PIL.ImageTk
import tensorflow as tf
import numpy as np

class Recon:
    def __init__(self, window, window_title):
        self.window = window
        self.window.title('Sign Recogniser')

        bg = tkinter.PhotoImage(file ="D:\SE project\machine learning model\slrecon9.png")
  
        label1 = tkinter.Label( self.window, image = bg)
        label1.place(x = -50, y = -60)
        self.window.bind('<Escape>', lambda e: window.quit())
        message = tkinter.Label(
            window, text="Sign Recognition System ",
            bg="white", fg="black", width=37,
            height=2, font=('Ariel', 25, 'bold'))
        message.place(x=400, y=700)

        
        self.vid = MyVideoCapture(0)

        self.canvas = tkinter.Canvas(window, width=self.vid.width, height=self.vid.height)
        self.canvas.pack()

        self.delay = 1
        self.update()
        self.window.mainloop()

    def update(self):

        ret,frame = self.vid.get_frame()
        if ret:
            self.photo = PIL.ImageTk.PhotoImage(image = PIL.Image.fromarray(frame))
            self.canvas.create_image(0, 0, image = self.photo, anchor = tkinter.NW)
    
        self.window.after(self.delay, self.update)

class MyVideoCapture:
    def __init__(self, video_source=0):
        self.vid = cv2.VideoCapture(video_source)

        if not self.vid.isOpened():
            raise ValueError("Unable to open video source", video_source)

        self.width = self.vid.get(cv2.CAP_PROP_FRAME_WIDTH)
        self.height = self.vid.get(cv2.CAP_PROP_FRAME_HEIGHT)
        

    def get_frame(self):
        if self.vid.isOpened():
            ret, frame = self.vid.read()
            
            frame=detect(frame)

        return ret,frame

    def __del__(self):
        if self.vid.isOpened():
            self.vid.release()



def decode(value):
    class_indices = {'A': 0, 'B': 1, 'C': 2, 'D': 3, 'E': 4, 'F': 5, 'G': 6, 'H': 7, 'I': 8, 'J': 9, 'K': 10,
                     'L': 11, 'M': 12, 'N': 13, 'O': 14, 'P': 15, 'Q': 16, 'R': 17, 'S': 18, 'T': 19, 'U': 20,
                     'V': 21, 'W': 22, 'X': 23, 'Y': 24, 'Z': 25, 'nothing': 26}

    key_list = list(class_indices.keys())
    val_list = list(class_indices.values())

    position = val_list.index(value)
    return key_list[position]

def detect(image):
    fontScale = 1

    color = (0, 0, 255)

    thickness = 4

    font = cv2.FONT_HERSHEY_SIMPLEX

    model = tf.keras.models.load_model('D:\SE project\machine learning model\sl_model.h5')

    resized = cv2.resize(image, (128,128))
    img_array = np.array([resized])
    prediction = model.predict(img_array)
    pred = decode(np.argmax(prediction))

    cv2.putText(image,pred,(100,100),font, fontScale,
                                color, thickness, cv2.LINE_AA, False)

    image=cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
    
    return image



Recon(tkinter.Tk(), "Tkinter and OpenCV")