def use(ai, input_tensors, show_logs=True):
    output_tensors = []
    for done, input_tensor in enumerate(input_tensors):
        output_tensors.append(ai.use(input_tensor))

        if show_logs:
            print('Done: ' + str(done * 100 / len(input_tensors)) + '%')
    return output_tensors
