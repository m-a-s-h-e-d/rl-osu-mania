from PIL import ImageGrab
import pyautogui
import cv2
import numpy as np
import matplotlib.pyplot as plt
import time

def resize_frame(frame, width, height):
    return cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)
    
screen_width, screen_height = pyautogui.size()
ratio = float(2.8)
left_padding = int(screen_width / ratio)
right_padding = screen_width - left_padding
bottom_height = screen_height - int(screen_height / 2)
time.sleep(5)
im = ImageGrab.grab(bbox =(left_padding, bottom_height, right_padding, screen_height)) 

window_width = right_padding - left_padding
window_height = screen_height - bottom_height
width = int(window_width * 0.07)
height = int(window_height * 0.07)

print(width, height)

shot = np.array(im)

gray = cv2.cvtColor(shot, cv2.COLOR_BGR2GRAY)

gray = resize_frame(gray, width, height)

plt.imshow(gray, cmap='gray')
plt.show()