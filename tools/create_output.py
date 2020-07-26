import json

input = './dataset/input/jet.json'
outut = './dataset/output/jet.json'


def create_item(input, index):
    return [1, 0]


file = open(input, 'r')
dataset = file.read()
file.close()

dataset = json.loads(dataset)
dataset_output = []
for index, item in enumerate(dataset):
    dataset_output.append(create_item(item, index))

file = open(outut, 'w')
file.write(json.dumps(dataset_output))
file.close()