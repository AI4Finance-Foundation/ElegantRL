class Args:
    def __init__(self):
        self.a = 1
        self.b = 2
        self.c = 3


def test(a, b, **_kwargs):
    # print(a, b, _kwargs)
    # print('c' in _kwargs)
    # print('d' in _kwargs)
    read(d=11, **vars(args))


def read(a, c=0, d=0, **_kwargs):
    print(a, c, d, _kwargs)


args = Args()
test(**vars(args))
