import cv2
import numpy as np



image = cv2.imread('dog.jpg')

image = cv2.resize(image, (224, 224))

image = image.astype(np.float32)

File = open('dog.bin', "wb")

File.write(image.tobytes())
