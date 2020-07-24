from app.layers.CONV import CONV
from app.layers.POOL import POOL
from app.layers.RELU import RELU
from app.layers.FC import FC

class AI:
    def __init__(self, config):
        self.input = config['input']
        self.output = config['output']

        self.layers = []
        for layer_data in config['layers']:
            layer = self.create_layer(layer_data['type'], layer_data['params'])

            if 'synapses' in layer_data:
                layer.set_synapses(layer_data['synapses'])

            self.layers.append(layer)

    def get_config(self):
        layers_config = []
        for layer in self.layers:
            layer_config = {
                'type': layer.__class__.__name__,
                'params': layer.params,
                'synapses': layer.get_synapses(),
            }
            layers_config.append(layer_config)

        config = {
            'input': self.input,
            'output': self.output,
            'layers': layers_config,
        }
        return config

    def set_hyper_params(self, hyper_params):
        for layer in self.layers:
            layer.set_hyper_params(hyper_params)
        return self

    def use(self, data):
        for layer in self.layers:
            data = layer.use(data)

        return data

    def train(self, delta):
        for layer in reversed(self.layers):
            delta = layer.train(delta)

    @staticmethod
    def create_layer(type, params):
        if type == 'CONV':
            return CONV(params)

        if type == 'POOL':
            return POOL(params)

        if type == 'RELU':
            return RELU(params)

        if type == 'FC':
            return FC(params)

        print('Error: unknown type of layer')

