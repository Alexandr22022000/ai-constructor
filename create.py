import json
from app.ai import AI
from app.scripts.files import write_file

config = {
    'layers': [
        {'type': "CONV", 'params': {'output_layers': 3, 'filter_size': [4, 4, 3], 'step': 2}},
        {'type': "RELU", 'params': {'func': 'tanh'}},
        {'type': "CONV", 'params': {'output_layers': 2, 'filter_size': [5, 5, 3], 'step': 2}},
        {'type': "RELU", 'params': {'func': 'tanh'}},
        {'type': "CONV", 'params': {'output_layers': 1, 'filter_size': [5, 5, 2], 'step': 2}},
        {'type': "RELU", 'params': {'func': 'tanh'}},

        {'type': "FC", 'params': {'input_neurons': [10, 10, 1], 'output_neurons': 100}},
        {'type': "RELU", 'params': {'func': 'tanh'}},
        {'type': "FC", 'params': {'input_neurons': 100, 'output_neurons': 50}},
        {'type': "RELU", 'params': {'func': 'tanh'}},
        {'type': "FC", 'params': {'input_neurons': 50, 'output_neurons': 10}},
        {'type': "RELU", 'params': {'func': 'tanh'}},
        {'type': "FC", 'params': {'input_neurons': 10, 'output_neurons': 2}},
        {'type': "RELU", 'params': {'func': 'tanh'}},

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
    'input': [100, 100, 3],
    'output': [1, 1, 2],
}

filename = 'ais/cool.ai'

config = AI(config).get_config()
write_file(filename, config)

print('AI was created and saved to ' + filename)
