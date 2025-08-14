# todo: add more model parameters, such as hidden layer size
def gen_attention_params(n, h=32, lr=0.001):
    return {'model': 'attention',
            'model_params': {'n': n,
                             'h': h,
                             'lr': lr}
            }
def gen_double_attention_params(n, h=32, lr=0.001):
    return {'model': 'double_attention',
            'model_params': {'n': n,
                             'h': h,
                             'lr': lr}
            }



def gen_dense_params(m, n, t, lr=0.001):
    return {'model': 'dense',
            'model_params': {'m': m,
                             'n': n,
                             't': t,
                             'lr': lr}
            }

def gen_rnn_params(n, lr):
    return {'model': 'rnn',
            'model_params': {'n': n,
                             'lr': lr}
            }


def gen_rand_params():
    return {'model': 'random',
            'model_params': {}
            }
