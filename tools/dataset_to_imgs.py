import cv2
import json
import numpy as np
from pathlib import Path

dataset = './dataset/output/car.json'
dir = './dataset/output/car/'

Path(dir).mkdir(parents=True, exist_ok=True)

file = open(dataset, 'r')
dataset = file.read()
file.close()

dataset = json.loads(dataset)
for index, img in enumerate(dataset):
    img = np.array(img)
    cv2.imwrite(dir + str(index) + '.jpg', img)




