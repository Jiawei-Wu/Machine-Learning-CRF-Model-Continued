import numpy as np

TrainSetSize = 120

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
print(vocab_x2[:100])
x2 = np.array(x)

y = [int(i.split(',')[2]) for i in y]
y = [y[i:i + 24] for i in range(len(y) / 24)]
print('\ny:')
print(y[:3])
y = np.array(y)

# st = np.stack([x1,x2,y])
st = np.stack([x2, y])
st = np.transpose(st, [1, 2, 0])
print('\nprint st shape:')
print(np.shape(st))

st = np.mean(st, 1, keepdims=False)
st = [(int(i), int(j)) for i, j in st]
print(np.shape(st))

train_st = st[:TrainSetSize]
test_st = st[TrainSetSize:]

set_st = set([i[0] for i in train_st])
vocab_st = sorted(set_st)
print('\nvocab_train_st:')
print(vocab_st)

proc_test = []
for i, j in test_st:
    if i in set_st:
        proc_test.append((i, j))
    else:
        for val in vocab_st:
            if val > i:
                proc_test.append((val, j))
                break

set_st = set([i[0] for i in proc_test])
vocab_st = sorted(set_st)
print('\nvocab_test_st:')
print(vocab_st)

for bi in train_st:
    tr_data.write(' '.join([str(i) for i in bi]) + '\n')

for bi in proc_test:
    te_data.write(' '.join([str(i) for i in bi]) + '\n')

tr_data.close()
te_data.close()
