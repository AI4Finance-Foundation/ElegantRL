
def gen_rnd_dense(m, n, t, lr=0.001):
    return {'model': 'dense',
            'model_params': {'m': m,
                             'n': n,
                             't': t,
                             'lr': lr},
            }


def gen_no_rnd():
    return {'model': 'None',
            'model_params': {},
            }
