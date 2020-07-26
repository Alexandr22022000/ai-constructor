import json
from app.ai import AI
from app.scripts.files import read_file, write_file
from app.scripts.use import use
from app.scripts.test import test
from app.scripts.train import train
from app.scripts.prepare_dataset import prepare_dataset

ai = 'cool.ai'
input = 'person.json'
output = 'person.json'

hyper_params = {
    'speed': 0.1,
    'moment': 0.01,
    'epochs': 1,
}

show_logs = True

config = read_file('ais/' + ai)
model = AI(config).set_hyper_params(hyper_params)
print('AI was loaded')

input_matrix = read_file('dataset/input/' + input)
output_matrix_ideal = read_file('dataset/output/' + output)
input_matrix = prepare_dataset(input_matrix, config['input'])
output_matrix_ideal = prepare_dataset(output_matrix_ideal, config['output'])

for epoch in range(hyper_params['epochs']):
    train(model, input_matrix, output_matrix_ideal, show_logs)
    output_matrix = use(model, input_matrix)
    error = test(output_matrix, output_matrix_ideal)
    print('Epoch was finished, error is ' + str(error * 100) + ' %')

write_file('ais/' + ai, model.get_config())
print('Train was finished')
