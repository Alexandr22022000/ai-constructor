import cv2
import json
from os import listdir

dir = './dataset/input/car_small/'
dataset = './dataset/input/car.json'
compression = (800, 600)

images = listdir(dir)
for key, image in enumerate(images):
    img = cv2.imread(dir + image)
    # img = cv2.resize(img, compression)
    images[key] = img.tolist()

file = open(dataset, 'w')
file.write(json.dumps(images))
file.close()
