for a in range(0,9):
    for b in range(0,9):
        for c in range(0,9):
            for d in range(0, 9):
                for e in range(0, 9):
                    for f in range(0, 9):
                        for g in range(0, 9):
                            for h in range(0, 9):
                                for i in range(0, 9):
                                    if a != b and a != c and a != d and a != e and a != f and a != g and a != h and a != i \
                                            and b != c and b != d and b != e and b != f and b != g and b != h and b != i\
                                                and c != d and c != e and c != f and c != g and c != h and c != i\
                                                    and d != e and d != f and d != g and d != h and d != i\
                                                        and e != f and e != g and e != h and e != i\
                                                            and f != g and f != h and f != i\
                                                                and g != h and g != i\
                                                                    and h != i:
                                        print('[', a, ',', b, ',', c, ',', d, ',', e, ',', f, ',', g, ',', h, ',', i, "]", end=', ')



