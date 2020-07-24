import json
from app.ai import AI
from app.scripts.files import write_file

config = {
    'layers': [
        {'type': "POOL", 'params': {'type': 'MID', 'compression': [1, 1]}},
        {'type': "CONV", 'params': {'output_layers': 2, 'filter_size': [2, 2, 2]}},
        {'type': "RELU", 'params': {'func': 'tanh'}},
        {'type': "CONV", 'params': {'output_layers': 2, 'filter_size': [2, 2, 2]}},
        {'type': "RELU", 'params': {'func': 'tanh'}},
        {'type': "FC", 'params': {'input_neurons': [2, 2, 2], 'output_neurons': [1, 1, 6]}},
        {'type': "RELU", 'params': {'func': 'tanh'}},
        {'type': "FC", 'params': {'input_neurons': [1, 1, 6], 'output_neurons': 1}},

        # {'type': "CONV", 'params': {'output_layers': 3, 'filter_size': [2, 2, 3]}},
        # {'type': "CONV", 'params': {'output_layers': 3, 'filter_size': [2, 2, 3]}},
        # {'type': "CONV", 'params': {'output_layers': 3, 'filter_size': [2, 2, 3]}},
        # {'type': "CONV", 'params': {'output_layers': 3, 'filter_size': [2, 2, 3]}},
        # {'type': "POOL", 'params': {'type': 'MID', 'compression': [10, 10]}}

        # {'type': "FC", 'params': {'input_neurons': 2, 'output_neurons': 3}},
        # {'type': "RELU", 'params': {'func': 'tanh'}},
        # {'type': "FC", 'params': {'input_neurons': 3, 'output_neurons': 1}},
        # {'type': "RELU", 'params': {'func': 'tanh'}},
    ],
    'input': [4, 4, 2],
    'output': [1, 1, 1],
}

filename = 'ais/cool.ai'

config = AI(config).get_config()
write_file(filename, config)

print('AI was created and saved to ' + filename)
