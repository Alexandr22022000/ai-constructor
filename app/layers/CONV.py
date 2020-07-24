import random
from ..Tensor import Tensor


class CONV:
    def __init__(self, params):
        if 'filter_size' not in params: params['filter_size'] = [2, 2, 1]
        if 'output_layers' not in params: params['output_layers'] = 1
        if 'step' not in params: params['step'] = 1
        if 'additional_pixels' not in params: params['additional_pixels'] = [0, 0]
        if 'displacement_neurons' not in params: params['displacement_neurons'] = True
        self.params = params

        self.speed = 1
        self.moment = 0.1
        self.last_input = None

        self.step = params['step']
        self.additional_pixels = params['additional_pixels']

        self.filters = []
        self.last_filters_deltas = []
        for i in range(params['output_layers']):
            self.filters.append(Tensor(params['filter_size']).random_values())
            self.last_filters_deltas.append(Tensor(params['filter_size']))

        self.displacement_neurons = None
        if params['displacement_neurons']:
            self.displacement_neurons = []
            self.last_displacement_neurons_deltas = [0] * params['output_layers']
            for i in range(params['output_layers']):
                self.displacement_neurons.append(random.uniform(-1, 1))

    def set_hyper_params(self, hyper_params):
        self.speed = hyper_params['speed']
        self.moment = hyper_params['moment']
        return self

    def set_synapses(self, synapses):
        self.filters = []
        for filter in synapses['filters']:
            self.filters.append(Tensor().set_matrix(filter))

        self.displacement_neurons = synapses['displacement_neurons']
        return self

    def get_synapses(self):
        filters = []
        for filter in self.filters:
            filters.append(filter.get_matrix())

        synapses = {
            'filters': filters,
            'displacement_neurons': self.displacement_neurons
        }
        return synapses

    def use(self, input_tensor):
        self.last_input = input_tensor.copy()

        steps_x = (input_tensor.size_x + self.additional_pixels[0] * 2 - self.filters[0].size_x) / self.step + 1
        steps_y = (input_tensor.size_y + self.additional_pixels[1] * 2 - self.filters[0].size_y) / self.step + 1

        if (steps_x % 1 != 0) or (steps_y % 1 != 0):
            print("Error: invalid size of data matrix")
            return

        steps_x = int(steps_x)
        steps_y = int(steps_y)

        output_tensor = Tensor(steps_x, steps_y, len(self.filters))

        # Each filter
        for output_layer_index in range(output_tensor.size_z):

            # Each filter step
            for step_x_index in range(steps_x):
                step_x = step_x_index * self.step - self.additional_pixels[0]
                for step_y_index in range(steps_y):
                    step_y = step_y_index * self.step - self.additional_pixels[1]

                    # Add displacement neuron
                    if self.displacement_neurons:
                        value = self.displacement_neurons[output_layer_index]
                        output_tensor.add(step_x_index, step_y_index, output_layer_index, value)

                    # Each pixel in filter
                    for filter_x in range(self.filters[0].size_x):
                        x = step_x + filter_x
                        for filter_y in range(self.filters[0].size_y):
                            y = step_y + filter_y

                            # If pixel in input tensor
                            if 0 < x < input_tensor.size_x and 0 < y < input_tensor.size_y:

                                # Each input layer
                                for z in range(input_tensor.size_z):
                                    value = input_tensor.get(x, y, z) * self.filters[output_layer_index].get(filter_x, filter_y, z)
                                    output_tensor.add(step_x_index, step_y_index, output_layer_index, value)

        return output_tensor

    def train(self, delta_tensor):
        if not self.last_input:
            print('Error: no input data for train!')
            return

        # ---GET GRADS---

        displacement_neurons_grads = [0] * len(self.filters)
        filters_grads = []
        for i in enumerate(self.filters):
            filters_grads.append(Tensor(self.filters[0].size_x, self.filters[0].size_y, self.filters[0].size_z))

        # Each filter
        for output_z in range(delta_tensor.size_z):

            # Each filter step
            for output_x in range(delta_tensor.size_x):
                step_x = output_x * self.step - self.additional_pixels[0]
                for output_y in range(delta_tensor.size_y):
                    step_y = output_y * self.step - self.additional_pixels[1]

                    if self.displacement_neurons is not None:
                        displacement_neurons_grads[output_z] += delta_tensor.get(output_x, output_y, output_z)

                    # Each pixel in filter
                    for filter_x in range(self.filters[0].size_x):
                        x = step_x + filter_x
                        for filter_y in range(self.filters[0].size_y):
                            y = step_y + filter_y

                            # If pixel in input tensor
                            if 0 < x < self.last_input.size_x and 0 < y < self.last_input.size_y:

                                # Each input layer
                                for z in range(self.last_input.size_z):
                                    value = self.last_input.get(x, y, z) * delta_tensor.get(output_x, output_y,
                                                                                            output_z)
                                    filters_grads[output_z].add(filter_x, filter_y, z, value)

        # ---UPDATE WEIGHTS----

        for key, filter in enumerate(self.filters):
            if self.displacement_neurons is not None:
                delta = displacement_neurons_grads[key] * self.speed
                self.displacement_neurons[key] += delta + self.last_displacement_neurons_deltas[key] * self.moment
                self.last_displacement_neurons_deltas[key] = delta

            for x in range(filter.size_x):
                for y in range(filter.size_y):
                    for z in range(filter.size_z):
                        delta = filters_grads[key].get(x, y, z) * self.speed
                        self.filters[key].add(x, y, z, delta + self.last_filters_deltas[key].get(x, y, z) * self.moment)
                        self.last_filters_deltas[key].set(x, y, z, delta)

        # ---GET OUTPUT DELTAS---

        output_delta = Tensor(self.last_input.size_x, self.last_input.size_y, self.last_input.size_z)
        input_delta_scaled = Tensor(self.step * (delta_tensor.size_x - 1) + 1,
                                    self.step * (delta_tensor.size_y - 1) + 1, delta_tensor.size_z)

        additional_pixels_reverse = [
            self.filters[0].size_x - 1 - self.additional_pixels[0],
            self.filters[0].size_y - 1 - self.additional_pixels[1],
        ]

        # Create scaled input delta tensor (like a step = 1)
        for x in range(delta_tensor.size_x):
            for y in range(delta_tensor.size_y):
                for z in range(delta_tensor.size_z):
                    input_delta_scaled.set(x * self.step, y * self.step, z, delta_tensor.get(x, y, z))

        # Each output delta pixel (input pixels) (reverse filter steps + filter layers)
        for x in range(output_delta.size_x):
            step_x = x - additional_pixels_reverse[0]
            for y in range(output_delta.size_y):
                step_y = y - additional_pixels_reverse[1]
                for z in range(output_delta.size_z):

                    # Each pixel in filter
                    for filter_x in range(self.filters[0].size_x):
                        output_x = step_x + filter_x
                        for filter_y in range(self.filters[0].size_y):
                            output_y = step_y + filter_y

                            # If pixel in delta tensor
                            if 0 < x < input_delta_scaled.size_x and 0 < y < input_delta_scaled.size_y:

                                # Each filter
                                for filter_index in range(input_delta_scaled.size_z):
                                    value = input_delta_scaled.get(output_x, output_y, filter_index)
                                    value *= self.filters[filter_index].get(self.filters[0].size_x - filter_x - 1,
                                                                            self.filters[0].size_y - filter_y - 1, z)
                                    output_delta.set(x, y, z, value)

        return output_delta
