import numpy as np
from sklearn import svm
from sklearn.neural_network import MLPClassifier
from sklearn.cluster import KMeans

x= np.asarray([[0,0],[1,0],[0,1],[1,1]])
y=np.asarray([0,1,1,0])
d=np.load('data4.npy')
t=np.load('targ4.npy')
D=np.abs(d);
d0=D[:,0]
d1=D[:,1]
d2=D[:,2]
d3=D[:,3]
d4=D[:,4]
d5=D[:,5]
d6=D[:,6]
data=np.log10(np.transpose(np.array([d0*d1,d2,d3,d4,d5,d6])))
mlp = MLPClassifier(hidden_layer_sizes=(100,10,100,100))
mlp.fit(data,t)
print(mlp.predict(data[22:27]))