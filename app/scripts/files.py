import json

def write_file(filename, data):
    file = open(filename, 'w')
    file.write(json.dumps(data))
    file.close()

def read_file(filename):
    file = open(filename, 'r')
    data = file.read()
    file.close()
    return json.loads(data)
