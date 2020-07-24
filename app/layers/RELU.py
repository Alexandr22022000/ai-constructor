from ..functions import functions, derivatives

class RELU:
    def __init__(self, params):
        if 'func' not in params: params['func'] = 'tanh'
        self.params = params

        self.last_input = None

        self.func = functions[params['func']]
        self.re_func = derivatives[params['func']]

    def set_hyper_params(self, hyper_params):
        return self

    def set_synapses(self, synapses):
        return self

    def get_synapses(self):
        return {}

    def use(self, input_tensor):
        input_tensor = input_tensor.copy()
        self.last_input = input_tensor.copy()

        for x in range(input_tensor.size_x):
            for y in range(input_tensor.size_y):
                for z in range(input_tensor.size_z):
                    input_tensor.set(x, y, z, self.func(input_tensor.get(x, y, z)))

        return input_tensor

    def train(self, delta_tensor):
        delta_tensor = delta_tensor.copy()
        if not self.last_input:
            print('Error: no input data for train!')
            return

        for x in range(delta_tensor.size_x):
            for y in range(delta_tensor.size_y):
                for z in range(delta_tensor.size_z):
                    delta_tensor.set(x, y, z, delta_tensor.get(x, y, z) * self.re_func(self.last_input.get(x, y, z)))

        return delta_tensor
