import cv2 as cv

def start(Buffer, Buffer_images, I):

    for i in range(0, 13):
        Buffer[str(i)] = [0, 0, 0]
        Buffer_images.append(I)
        name = 'b' + str(i) + '.jpg'
        cv.imwrite(name, I)
    return Buffer, Buffer_images