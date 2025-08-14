"""
returns the configuration for a specific critic
"""
def gen_critic_dense(m, n, t, lr=0.001):
    return {'model': 'dense',
            'model_params': {'m': m,
                             'n': n,
                             't': t,
                             'lr': lr}
            }

def gen_no_critic():
    return {'model' :'None',
            'model_params' : { }
            }
