import cv2
import json
from os import listdir

dir = './dataset/input/jet/'
dataset = './dataset/input/jet.json'
compression = (100, 100)

images = listdir(dir)
dataset_array = []
for key, image in enumerate(images):
    img = cv2.imread(dir + image)
    if img is None: continue
    img = cv2.resize(img, compression)
    dataset_array.append(img.tolist())

file = open(dataset, 'w')
file.write(json.dumps(dataset_array))
file.close()
