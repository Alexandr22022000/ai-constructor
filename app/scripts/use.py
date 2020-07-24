def use(ai, input_tensors):
    output_tensors = []
    for input_tensor in input_tensors:
        output_tensors.append(ai.use(input_tensor))
    return output_tensors
