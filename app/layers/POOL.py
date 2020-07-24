from ..Tensor import Tensor


class POOL:
    def __init__(self, params):
        if 'type' not in params: params['type'] = POOL_TYPE.MID
        if 'compression' not in params: params['compression'] = [2, 2]
        self.params = params

        self.last_input = None
        self.type = params['type']
        self.compression = params['compression']

    def set_hyper_params(self, hyper_params):
        return self

    def set_synapses(self, synapses):
        return self

    def get_synapses(self):
        return {}

    def use(self, input_tensor):
        self.last_input = input_tensor.copy()

        steps_x = input_tensor.size_x / self.compression[0]
        steps_y = input_tensor.size_y / self.compression[1]

        if (steps_x % 1 != 0) or (steps_y % 1 != 0):
            print("Error: invalid size of data matrix")
            return

        steps_x = int(steps_x)
        steps_y = int(steps_y)

        output_tensor = Tensor(steps_x, steps_y, input_tensor.size_z)

        # Each layer
        for z in range(input_tensor.size_z):

            # Each filter step
            for step_x_index in range(steps_x):
                step_x = step_x_index * self.compression[0]
                for step_y_index in range(steps_y):
                    step_y = step_y_index * self.compression[1]

                    values = []

                    # Each pixel in filter
                    for filter_x in range(self.compression[0]):
                        x = step_x + filter_x
                        for filter_y in range(self.compression[1]):
                            y = step_y + filter_y

                            values.append(input_tensor.get(x, y, z))

                    output_tensor.set(step_x_index, step_y_index, z, self.calc_output_pixel_value(values))
        return output_tensor

    def train(self, delta_tensor):
        if not self.last_input:
            print('Error: no input data for train!')
            return

        output_delta = Tensor(self.last_input.size_x, self.last_input.size_y, self.last_input.size_z)

        # Each layer
        for z in range(self.last_input.size_z):

            # Each filter step
            for step_x_index in range(delta_tensor.size_x):
                step_x = step_x_index * self.compression[0]
                for step_y_index in range(delta_tensor.size_y):
                    step_y = step_y_index * self.compression[1]

                    input_bit = []
                    for x in range(self.compression[0]):
                        input_bit.append([])
                        for y in range(self.compression[1]):
                            input_bit[x].append(0)

                    # Each pixel in filter (get input matrix bit)
                    for filter_x in range(self.compression[0]):
                        x = step_x + filter_x
                        for filter_y in range(self.compression[1]):
                            y = step_y + filter_y

                            input_bit[filter_x][filter_y] = self.last_input.get(x, y, z)

                    delta_bit = self.calc_delta_values(delta_tensor.get(step_x_index, step_y_index, z), input_bit)

                    # Each pixel in filter (set output delta matrix bit)
                    for filter_x in range(self.compression[0]):
                        x = step_x + filter_x
                        for filter_y in range(self.compression[1]):
                            y = step_y + filter_y

                            output_delta.set(x, y, z, delta_bit[filter_x][filter_y])

        return output_delta

    def calc_output_pixel_value(self, pixels):
        if self.type == 'MAX':
            max = 0
            for pixel in pixels:
                if pixel > max:
                    max = pixel
            return max

        if self.type == 'MIN':
            min = 9999999999999
            for pixel in pixels:
                if pixel < min:
                    min = pixel
            return min

        if self.type == 'MID':
            mid = 0
            for pixel in pixels:
                mid += pixel
            mid /= len(pixels)
            return mid

        if self.type == 'SUM':
            sum = 0
            for pixel in pixels:
                sum += pixel
            return sum

    def calc_delta_values(self, delta, input_matrix):
        output_matrix = [[0] * len(input_matrix[0]) for x, v in enumerate(input_matrix)]

        if self.type == 'MAX':
            max = 0
            max_indexes = [0, 0]

            for x in range(len(input_matrix)):
                for y in range(len(input_matrix[0])):
                    if max < input_matrix[x][y]:
                        max = input_matrix[x][y]
                        max_indexes = [x, y]

            output_matrix[max_indexes[0]][max_indexes[1]] = delta
            return output_matrix

        if self.type == 'MIN':
            min = 0
            min_indexes = [0, 0]

            for x in range(len(input_matrix)):
                for y in range(len(input_matrix[0])):
                    if min > input_matrix[x][y]:
                        min = input_matrix[x][y]
                        min_indexes = [x, y]

            output_matrix[min_indexes[0]][min_indexes[1]] = delta
            return output_matrix

        if self.type == 'MID':
            value = delta / (len(input_matrix[0]) * len(input_matrix))
            output_matrix = [[value] * len(input_matrix[0])] * len(input_matrix)
            return output_matrix

        if self.type == 'SUM':
            output_matrix = [[delta] * len(input_matrix[0])] * len(input_matrix)
            return output_matrix



