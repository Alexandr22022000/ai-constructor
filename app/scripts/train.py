def train(ai, input, output_ideal):
    for key, input_tensor in enumerate(input):
        output_tensor = ai.use(input_tensor)

        for x in range(output_tensor.size_x):
            for y in range(output_tensor.size_y):
                for z in range(output_tensor.size_z):
                    value = output_ideal[key].get(x, y, z) - output_tensor.get(x, y, z)
                    output_tensor.set(x, y, z, value)

        ai.train(output_tensor)
