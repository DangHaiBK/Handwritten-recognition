import cv2
import numpy as np
from tkinter import *
from PIL import ImageGrab
from keras.models import load_model

# Load .h5 model to predict
model = load_model('D:/Models/mnist_model.h5')

# Create the main window for App
root = Tk()
root.resizable(0, 0)
root.title("Handwritten Recognition Application")

# Initialize variables
initx, inity = None, None
image_number = 0


# Delete the current content inside Canvas
def clear_source():
    global draw_area
    draw_area.delete("all")  # Delete method for cleaning


# Event occurrence when clicking the left mouse
def activate_event(event):
    global initx, inity
    draw_area.bind('<B1-Motion>', draw_lines)  # Session has started and call draw_lines
    initx, inity = event.x, event.y


# Draw lines inside Canvas
def draw_lines(event):
    global initx, inity
    x, y = event.x, event.y
    draw_area.create_line((initx, inity, x, y), width=7, fill='black', capstyle=ROUND, smooth=True, splinesteps=12)
    initx, inity = x, y


# Make one or more predictions from the loaded model
def Recognize_Digit():
    global image_number
    filename = f'image_{image_number}.png'
    widget = draw_area

    # Get coordinates of Canvas
    x = root.winfo_rootx() + widget.winfo_x()
    y = root.winfo_rooty() + widget.winfo_y()
    x1 = x + widget.winfo_width()
    y1 = y + widget.winfo_height()

    # Get image by using grab() and crop it. Then save it into PIL (temporary)
    ImageGrab.grab().crop((x, y, x1, y1)).save(filename)

    # Read the saved image from PIL with full color
    digit = cv2.imread(filename, cv2.IMREAD_COLOR)

    # Convert it into grayscale
    make_gray = cv2.cvtColor(digit, cv2.COLOR_BGR2GRAY)

    # Create an OTSU threshold before finding contours to get better result
    ret, th = cv2.threshold(make_gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # Find contours of the image
    contours = cv2.findContours(th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]

    for cnt in contours:
        # Get bounding box and extract the region of interest (ROI)
        x, y, w, h = cv2.boundingRect(cnt)

        # Create Rectangle
        cv2.rectangle(digit, (x, y), (x + w, y + h), (255, 0, 0), 1)
        top = int(0.05 * th.shape[0])
        bottom = top
        left = int(0.05 * th.shape[1])
        right = left

        # Extract the image ROI
        roi = th[y - top:y + h + bottom, x - left:x + w + right]

        # Resize ROI image to 28*28 pixels
        img = cv2.resize(roi, (28, 28), interpolation=cv2.INTER_AREA)

        # Reshape image to support the input of the model
        # img=img.astype('float32')
        img = img.reshape(1, 28 * 28) # -- For MLP model
        #img = img.reshape(1, 28, 28, 1) # -- For CNN model
        # Normalize image to support the requirement for the model's input
        img = img / 255.0
        #print(img)
        prediction = model.predict([img])[0]

        # Get the maximum values
        final = np.argmax(prediction)
        data = str(final) + '  ' + str(int(max(prediction) * 100)) + '%'

        # Put a text string on image: font, font_scale, color, and thickness
        cv2.putText(digit, data, (x,y-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)

    # Show the predict results on a new window
    cv2.imshow('digit', digit)
    cv2.waitKey(0)


# Creating a Canvas for drawing
draw_area = Canvas(root, width=640, height=480, bg='white')
draw_area.grid(row=0, column=0, pady=2, sticky=W, columnspan=2)

# Mechanism to let you deal with event yourself
draw_area.bind('<Button-1>', activate_event)

# Add buttons and their functions
btn_save = Button(text="Recognize", fg='black', command=Recognize_Digit)
btn_save.grid(row=2, column=0, pady=1, padx=1)
button_clear = Button(text="Clear", fg='black', command=clear_source)
button_clear.grid(row=2, column=1, pady=1, padx=1)

# Application is ready to run
root.mainloop()