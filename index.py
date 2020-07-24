from os import listdir
import cv2

s = 'ssss'

if s[-1] != '/':
    s += '/'

print(s)

files = listdir('./dataset/input')

print(files)


img = cv2.imread('./dataset/input/cat.png')
print(img[1300][1300])
cv2.imwrite('./dataset/input/cat2.jpg', img)
