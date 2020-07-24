from ..Tensor import Tensor
import random


class FC:
    def __init__(self, params):
        if 'input_neurons' not in params: params['input_neurons'] = 2
        if 'output_neurons' not in params: params['output_neurons'] = 1
        if 'displacement_neurons' not in params: params['displacement_neurons'] = True
        self.params = params

        self.last_input = None
        self.speed = 1
        self.moment = 0.1

        self.input = params['input_neurons']
        if isinstance(self.input, (int, float)):
            self.input = [1, 1, self.input]
        input = self.input[0] * self.input[1] * self.input[2]

        self.output = params['output_neurons']
        if isinstance(self.output, (int, float)):
            self.output = [1, 1, self.output]
        output = self.output[0] * self.output[1] * self.output[2]

        self.synapses = [[random.uniform(-1, 1) for y in range(input)] for x in range(output)]
        self.last_synapses_deltas = [[0 for y in range(input)] for x in range(output)]

        self.displacement_neurons = None
        if params['displacement_neurons']:
            self.displacement_neurons = [random.uniform(-1, 1) for x in range(output)]
            self.last_displacement_neurons_deltas = [0] * output

    def set_hyper_params(self, hyper_params):
        self.speed = hyper_params['speed']
        self.moment = hyper_params['moment']
        return self

    def set_synapses(self, synapses):
        self.synapses = synapses['synapses']
        self.displacement_neurons = synapses['displacement_neurons']
        return self

    def get_params(self):
        params = {
            'input_neurons': len(self.synapses[0]),
            'output_neurons': len(self.synapses),
            'displacement_neurons': self.displacement_neurons is not None,
        }
        return params

    def get_synapses(self):
        synapses = {
            'synapses': self.synapses,
            'displacement_neurons': self.displacement_neurons,
        }
        return synapses

    def use(self, input_tensor):
        self.last_input = input_tensor.copy()

        input_array = input_tensor.to_array()
        output_array = [0] * len(self.synapses)

        for output_index, v in enumerate(self.synapses):
            output_array[output_index] += self.displacement_neurons[output_index]

            for input_index, v in enumerate(input_array):
                output_array[output_index] += input_array[input_index] * self.synapses[output_index][input_index]

        return Tensor().add_array(output_array).set_size(self.output)

    def train(self, delta_tensor):
        if not self.last_input:
            print('Error: no input to train!')
            return

        last_input_array = self.last_input.to_array()
        delta_array = delta_tensor.to_array()

        # ---GET OUTPUT DELTAS---

        output_deltas = [0] * len(self.synapses[0])

        for output_index, v in enumerate(self.synapses):
            for input_index, v in enumerate(self.synapses[0]):
                output_deltas[input_index] += delta_array[output_index] * self.synapses[output_index][input_index]

        # ---UPDATE WEIGHTS---

        for output_index, v in enumerate(self.synapses):
            if self.displacement_neurons is not None:
                self.displacement_neurons[output_index] += delta_array[output_index] * self.speed + self.last_displacement_neurons_deltas[output_index] * self.moment
                self.last_displacement_neurons_deltas[output_index] = delta_array[output_index]

            for input_index, v in enumerate(self.synapses[0]):
                delta = delta_array[output_index] * last_input_array[input_index]
                self.synapses[output_index][input_index] += delta * self.speed + self.last_synapses_deltas[output_index][input_index] * self.moment
                self.last_synapses_deltas[output_index][input_index] = delta

        return Tensor().add_array(output_deltas).set_size(self.input)
