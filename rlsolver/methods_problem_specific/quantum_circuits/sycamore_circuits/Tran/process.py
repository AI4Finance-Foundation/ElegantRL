# Convert the original order into a verifiable order
import numpy as np
with open('myhashmap.txt', 'r') as f:
    hashmap = f.readline()
    alphas = hashmap.split(',')
    
orders = []
lefts = []
rights = []
# read lines from current file
with open('mycurrent.txt', 'r') as f:
    lines = f.readlines()
    for line in lines:
        # each line has a format of "a , b -> c"
        # split the line into three parts
        ops, ans = line.replace('\n','').split('->')
        a, b = ops.split(',')
        try:
            index_a = alphas.index(a)
        except:
            try:
                ii = lefts.index(a)
                print('a found in lefts')
            except:
                try:
                    ii = rights.index(a)
                    print('a found in rights')
                except:
                    print('a:##{}## not found'.format(a))
        try:
            index_b = alphas.index(b)
        except:
            try:
                ii = lefts.index(b)
                print('b found in lefts')
            except:
                try:
                    ii = rights.index(b)
                    print('b found in rights')
                except:
                    print('b:##{}## not found'.format(b))
        orders.append((index_a, index_b))
        alphas[index_a] = ans
        alphas[index_b] = ans
        lefts.append(a)
        lefts.append(b)
        rights.append(ans)
        
import pickle as pkl
with open('myorders.pkl', 'wb') as f:
    pkl.dump(orders, f)


# load orders
with open('myorders.pkl', 'rb') as f:
    orders = pkl.load(f)
    print(orders)   
