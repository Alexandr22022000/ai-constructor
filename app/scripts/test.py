def test(output, output_ideal):
    error = 0
    for key, output_item in enumerate(output):
        for x in range(output_item.size_x):
            for y in range(output_item.size_y):
                for z in range(output_item.size_z):
                    error += (output_ideal[key].get(x, y, z) - output_item.get(x, y, z)) ** 2
    error = error ** 0.5
    return error
