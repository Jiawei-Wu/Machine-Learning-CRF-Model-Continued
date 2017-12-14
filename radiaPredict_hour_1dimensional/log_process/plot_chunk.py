import matplotlib.pyplot as plt 
import numpy.linalg as la
import numpy as np 


te = open('test_log.txt')
tags = []
preds = []
for line in te:
    if len(line)<2:
        continue
    _, tag, pred = line.split('\t')
    tags.append(tag)
    preds.append(pred[:-1])
    
    
buf = []
print(zip(tags, preds))
for i, j in zip(tags, preds):
	if i == j:
		buf.append(0)
	else:
		buf.append(1)
print(buf)
buf = buf[:1000]
print('Test error curve (%):')
x = range(len(buf))
# print(zip(tags,preds))
plt.plot(x, buf)
plt.show()

