import cv2
import numpy as np
import pyttsx3
from keras.models import load_model

# Load the model
model = load_model("keras_model.h5", compile=False)

# Load the labels
class_names = open("labels.txt", "r").readlines()

# Load the pre-trained face detector
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

# Initialize pyttsx3 engine
engine = pyttsx3.init()

# CAMERA can be 0 or 1 based on the default camera of your computer
camera = cv2.VideoCapture(0)

while True:
    # Grab the webcam's image.
    ret, image = camera.read()

    # Convert the image to grayscale for face detection
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Detect faces in the image
    faces = face_cascade.detectMultiScale(gray_image, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    # Iterate over each detected face
    for (x, y, w, h) in faces:
        # Draw a rectangle around the face
        cv2.rectangle(image, (x, y), (x+w, y+h), (255, 0, 0), 2)

        # Extract the face region and preprocess for the model
        face_roi = gray_image[y:y+h, x:x+w]
        face_roi = cv2.resize(face_roi, (224, 224))
        # Convert the grayscale face region to a 3-channel image
        face_roi = cv2.cvtColor(face_roi, cv2.COLOR_GRAY2RGB)
        face_roi = np.asarray(face_roi, dtype=np.float32).reshape(1, 224, 224, 3)
        face_roi = (face_roi / 127.5) - 1

        # Predict the model
        prediction = model.predict(face_roi)
        index = np.argmax(prediction)
        class_name = class_names[index]
        confidence_score = prediction[0][index]

        # Write the class name and confidence score on the frame
        cv2.putText(image, f'{class_name[2:]}: {confidence_score:.2f}', (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        # Speak the result
        engine.say(class_name[2:])
        engine.runAndWait()

    # Show the image in a window
    cv2.imshow("Webcam Image", image)

    # Listen to the keyboard for presses.
    keyboard_input = cv2.waitKey(1)

    if keyboard_input == 2:
        break

camera.release()
cv2.destroyAllWindows()
