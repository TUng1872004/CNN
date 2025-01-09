import cv2
import numpy as np

def detect(im_path):
    face_cascade = cv2.CascadeClassifier(r'C:\My Computer\BachKhoa\HK5\ML\CNN\haarcascade_frontalface_default.xml')
    image = cv2.imread(im_path)
    faces = face_cascade.detectMultiScale(image, scaleFactor=1.1, minNeighbors=5)
    # If no faces are detected, return the original image
    if len(faces) == 0:
        return image
    # Find the largest face by area
    largest_face = max(faces, key=lambda face: face[2] * face[3])
    x, y, w, h = largest_face
    height, width, _ = image.shape
    x_start = max(0, x - 20)
    y_start = max(0, y - 20)
    x_end = min(width, x + w + 20)
    y_end = min(height, y + h + 20)

    face_crop = image[y_start:y_end, x_start:x_end]
    # Optional: Draw a rectangle around the largest detected face in the original image
    #cv2.rectangle(face_resized, (x, y), (x+w, y+h), (255, 0, 0), 2)
    return face_crop
