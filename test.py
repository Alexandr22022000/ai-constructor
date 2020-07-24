from app.ai import AI
from app.scripts.files import read_file
from app.scripts.use import use
from app.scripts.test import test
from app.scripts.prepare_dataset import prepare_dataset

ai = 'ais/cool.ai'
input = 'dataset/input/car.json'
output = 'dataset/output/car.json'

config = read_file(ai)
ai = AI(config)
print('AI was loaded')

input_matrix = read_file(input)
output_matrix_ideal = read_file(output)
input_matrix = prepare_dataset(input_matrix, config['input'])
output_matrix_ideal = prepare_dataset(output_matrix_ideal, config['output'])

print('Start processing data')
output_matrix = use(ai, input_matrix)
error = test(output_matrix, output_matrix_ideal)
print('Data processing is ended, error is ' + str(error * 100) + ' %')
