import random

class Tensor:
    def __init__(self, size_x=0, size_y=0, size_z=0):
        if not isinstance(size_x, (int, float)):
            self.size_x = size_x[0]
            self.size_y = size_x[1]
            self.size_z = size_x[2]
        else:
            self.size_x = size_x
            self.size_y = size_y
            self.size_z = size_z

        self.layers = [[[0 for z in range(self.size_z)] for y in range(self.size_y)] for x in range(self.size_x)]

    def set_size(self, size):
        array = self.to_array()

        self.size_x = size[0]
        self.size_y = size[1]
        self.size_z = size[2]

        def get_element():
            element = array[-1]
            del array[-1]
            return element

        self.layers = [[[0 for z in range(self.size_z)] for y in range(self.size_y)] for x in range(self.size_x)]
        for z in range(self.size_z - 1, -1, -1):
            for y in range(self.size_y - 1, -1, -1):
                for x in range(self.size_x - 1, -1, -1):
                    self.layers[x][y][z] = get_element()
        return self

    def set_matrix(self, matrix):
        self.size_x = len(matrix)
        self.size_y = len(matrix[0])
        self.size_z = len(matrix[0][0])
        self.layers = matrix
        return self

    def random_values(self):
        for x in range(self.size_x):
            for y in range(self.size_y):
                for z in range(self.size_z):
                    self.layers[x][y][z] = random.uniform(-1, 1)
        return self

    def add_invert_matrix(self, matrix):
        self.size_x = len(matrix[0][0])
        self.size_y = len(matrix[0])
        self.size_z = len(matrix)

        self.layers = [[[0 for z in range(self.size_z)] for y in range(self.size_y)] for x in range(self.size_x)]

        for z in range(self.size_z):
            for y in range(self.size_y):
                for x in range(self.size_x):
                    self.layers[x][y][z] = matrix[z][y][x]
        return self

    def add_array(self, array):
        self.size_x = 1
        self.size_y = 1
        self.size_z = len(array)

        self.layers = [[array]]
        return self

    def set(self, x, y, z, value):
        self.layers[x][y][z] = value

    def get(self, x, y, z):
        return self.layers[x][y][z]

    def add(self, x, y, z, value):
        self.layers[x][y][z] += value

    def sub(self, x, y, z, value):
        self.layers[x][y][z] -= value

    def to_array(self):
        array = []
        for z in range(self.size_z):
            for y in range(self.size_y):
                for x in range(self.size_x):
                    array.append(self.get(x, y, z))
        return array

    def get_matrix(self):
        return self.layers

    def copy(self):
        new_matrix = [[[0] * self.size_z] * self.size_y] * self.size_x
        for x in range(self.size_x):
            for y in range(self.size_y):
                for z in range(self.size_z):
                    new_matrix[x][y][z] = self.layers[x][y][z]

        return Tensor().set_matrix(new_matrix)




