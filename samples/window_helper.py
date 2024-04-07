from PIL import ImageGrab
import pyautogui
    
screen_width, screen_height = pyautogui.size()
ratio = float(2.8)
left_padding = round(screen_width / ratio)
right_padding = screen_width - left_padding
im2 = ImageGrab.grab(bbox =(left_padding, 0, right_padding, screen_height)) 

print(screen_height)
print(right_padding - left_padding)

im2.show()