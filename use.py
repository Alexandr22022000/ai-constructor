from app.ai import AI
from app.scripts.files import write_file, read_file
from app.scripts.use import use
from app.scripts.prepare_dataset import prepare_dataset

ai = 'ais/cool.ai'
input = 'dataset/input/car.json'
output = 'dataset/output/car2.json'
show_logs = True

config = read_file(ai)
ai = AI(config)
print('AI was loaded')

input_matrix = read_file(input)
input_matrix = prepare_dataset(input_matrix, config['input'])

print('Start processing data')
output_matrix = use(ai, input_matrix, show_logs)
for key, tensor in enumerate(output_matrix):
    output_matrix[key] = tensor.get_matrix()
write_file(output, output_matrix)
print('Data processing is ended, results saved to ' + output)
