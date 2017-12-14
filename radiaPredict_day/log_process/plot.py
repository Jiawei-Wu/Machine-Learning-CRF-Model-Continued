import matplotlib.pyplot as plt 
import numpy.linalg as la
import numpy as np 

tr = open('train_log.txt')

buf = []
for line in tr:
    if line[:4] == 'iter':
        arr = line.split(' ')
        buf.append(float(arr[4][4:]))

print('Training curve:')
y = buf
# print(buf)
x = range(len(buf))
plt.plot(x, y)
plt.show()

te = open('test_log.txt')
tags = []
preds = []
for line in te:
    if len(line)<2:
        continue
    _, tag, pred = line.split('\t')
    
    tags.append(float(tag))
    preds.append(float(pred))

print('Testing curve:')
x = range(len(tags))
plt.plot(x, preds, 'g', x, tags, 'b')
plt.show()

print('Test error curve (\%):')
x = range(len(tags))
ploty = [float(abs(ta-pre)) / pre * 100 for ta, pre in zip([tags,preds])]
plt.plot(x, preds, 'g', x, tags, 'b')
plt.show()


decode_float_vec = np.array(preds)
radia_float_vec = np.array(tags)

RMSE = la.norm(decode_float_vec-radia_float_vec,2)/np.sqrt(len(decode_float_vec))
MAPE = np.mean(abs((decode_float_vec-radia_float_vec)/radia_float_vec)*100)
MABE = la.norm(decode_float_vec-radia_float_vec,1)/len(decode_float_vec)

print('RMSE: %f\nMAPE: %f\nMABE: %f\n' % (RMSE,MAPE,MABE))
