import cv2
import numpy as np

def Resize(image, height, width):
    '''
    desired_size = (height, width)
    old_size = image.shape[:2]
    max_size_idx = old_size.index(max(old_size))
    ratio = float(desired_size[max_size_idx]) / max(old_size)
    new_size = tuple([int(x * ratio) for x in old_size])

    if new_size > desired_size:
        min_size_idx = old_size.index(min(old_size))
        ratio = float(desired_size[min_size_idx]) / min(old_size)
        new_size = tuple([int(x * ratio) for x in old_size])
    '''
    image = cv2.resize(image, (width, height))
    #delta_w = desired_size[1] - new_size[1]
    #delta_h = desired_size[0] - new_size[0]
    #top, bottom = delta_h // 2, delta_h - (delta_h // 2)
    #left, right = delta_w // 2, delta_w - (delta_w // 2)
    #image = cv2.copyMakeBorder(image, top, bottom, left, right, cv2.BORDER_CONSTANT)
    return image
    