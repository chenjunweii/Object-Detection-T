import json
import yaml
import pickle
import numpy as np


def load_json(filename):
    with open(filename, 'r') as f:
        #return yaml.safe_load(f)
        return json.load(f)

def save_json(dictionary, filename):
    with open(filename, 'w') as f:
        json.dump(dictionary, f)

def load_pickle(filename):
    with open(filename, 'rb') as f:
        return pickle.load(f)

def save_pickle(obj, filename):
    with open(filename, 'wb') as f:
        pickle.dump(obj, f)
    
if __name__ == '__main__':
    
    pass

def assign(g, node, data):
    data = np.array(data) if type(data) != np.ndarray else data
    if node in list(g.keys()):
        if g[node][...].shape == data.shape:
            g[node][...] = data
        else:
            del g[node]
            g[node] = data
    else:
        g[node] = data

