import cv2


def display(img, name, resizable=True):
    if resizable:
        cv2.namedWindow(name, cv2.WINDOW_NORMAL)
    cv2.imshow(name, img)