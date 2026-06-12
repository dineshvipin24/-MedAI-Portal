import numpy as np
from PIL import Image, ImageFilter

def generate_synthetic_xray():
    size = 512
    img = np.zeros((size, size), dtype=np.uint8)
    y, x = np.ogrid[:size, :size]
    img = (x * 0.05 + y * 0.08).astype(np.uint8)
    img = np.clip(img + 45, 45, 95)
    left_lung = ((x - 170)/65)**2 + ((y - 240)/140)**2 <= 1
    right_lung = ((x - 342)/65)**2 + ((y - 240)/140)**2 <= 1
    img[left_lung] = 22
    img[right_lung] = 22
    heart = ((x - 235)/85)**2 + ((y - 300)/70)**2 <= 1
    img[heart] = 125
    spine = (x >= 251) & (x <= 261) & (y >= 40) & (y <= 460)
    img[spine] = 155
    for rib_y in range(110, 420, 32):
        rib_left = (np.abs((y - rib_y) - (x - 170)**2/320) < 5) & (x < 251) & (x > 80)
        rib_right = (np.abs((y - rib_y) - (x - 342)**2/320) < 5) & (x > 261) & (x < 432)
        img[rib_left] = 100
        img[rib_right] = 100
    Image.fromarray(img).filter(ImageFilter.GaussianBlur(radius=3)).save('chest_test.png')

def generate_synthetic_brain():
    size = 512
    img = np.zeros((size, size), dtype=np.uint8)
    y, x = np.ogrid[:size, :size]
    skull_outer = ((x - 256)/180)**2 + ((y - 256)/210)**2 <= 1
    skull_inner = ((x - 256)/168)**2 + ((y - 256)/198)**2 <= 1
    img[skull_outer] = 180
    img[skull_inner] = 30
    brain_left = (((x - 175)/90)**2 + ((y - 256)/160)**2 <= 1) & skull_inner
    brain_right = (((x - 337)/90)**2 + ((y - 256)/160)**2 <= 1) & skull_inner
    img[brain_left] = 85
    img[brain_right] = 85
    ventricle_l = (((x - 230)/25)**2 + ((y - 230)/45)**2 <= 1) | (((x - 215)/15)**2 + ((y - 270)/30)**2 <= 1)
    ventricle_r = (((x - 282)/25)**2 + ((y - 230)/45)**2 <= 1) | (((x - 297)/15)**2 + ((y - 270)/30)**2 <= 1)
    img[ventricle_l & skull_inner] = 15
    img[ventricle_r & skull_inner] = 15
    for radius in range(50, 160, 22):
        sulci = (np.abs(((x - 256)**2 + (y - 256)**2) - radius**2) < 4) & (y % 15 < 6) & skull_inner
        img[sulci] = 55
    Image.fromarray(img).filter(ImageFilter.GaussianBlur(radius=4)).save('brain_test.png')

generate_synthetic_xray()
generate_synthetic_brain()
print("Mock images generated successfully!")
