import numpy as np

StartIndex = 100
TestNum = 10

PoolingSize_x1 = 3
PoolingSize_x2 = 3
PoolingSize_y = 50

x1 = open('raw_x1.csv')
x2 = open('raw_x2.csv')
y = open('raw_tags.csv')
tr_data = open('../data/train.data', 'w')
te_data = open('../data/test.data', 'w')


def delcol(fx):
    fx = fx.readlines()
    print('\nDebug1:')
    print(fx[:2])
    x = [[int(float(i)) for i in x.split(',')[1:]] for x in fx]
    print('\nDebug2:')
    print(x[:2])
    return x

# x = delcol(x1)
# vocab_x1 = sorted(set([i for j in x for i in j]))
# print('\nvocab_x1:')
# print(vocab_x1[:100])
# x1 = np.array(x)


x = delcol(x2)
x = [[sum(singx[i:i + 4]) // 4 for i in range(len(singx) / 4)] for singx in x]
vocab_x2 = sorted(set([i for j in x for i in j]))
print('\nvocab_x2:')
print(vocab_x2)
pool_x = [[i - (i % PoolingSize_x2)for i in j] for j in x]
x2 = np.array(pool_x)
print(x2)

y = [int(i.split(',')[2]) for i in y]
y = [y[i:i + 24] for i in range(len(y) / 24)]
pool_y = [[i - (i % PoolingSize_y)for i in j] for j in y]
y = np.array(pool_y)
print('\ny:')
print(y[:3])

st = np.stack([x2, y])
st = np.transpose(st, [1, 2, 0])
print('\nprint st shape:')
print(np.shape(st))
for index, s in enumerate(st):
    # r = np.random.rand(1)
    for tri in s:
        stri = [str(i) for i in tri]
        if index >= StartIndex and index < StartIndex + TestNum:
            te_data.write(' '.join(list(stri)) + '\n')
        else:
            tr_data.write(' '.join(list(stri)) + '\n')

    if index >= StartIndex and index < StartIndex + TestNum:
        te_data.write('\n')
    else:
        tr_data.write('\n')

tr_data.close()
te_data.close()
