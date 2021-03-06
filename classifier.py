import numpy as np
from sklearn.neural_network import MLPClassifier
class MLP:
    def __init__(self,md=''):
        if md=='':
            d=np.load('data5.npy')
            t=np.load('targ5.npy')
            tr=[]
            k=0
            for a in t:
                if k%2:
                    tr.append(a)
                k=k+1
            t=tr
            dt1=np.log10(D[:,0]*D[:,1])
            dt2=np.log10(D[:,2])
            dt3=np.log10(D[:,3])
            dt4=np.log10(D[:,4])
            dt5=np.log10(D[:,5])
            dt6=np.log10(D[:,6])
            dt7=(D[:,7])
            dt8=(D[:,8])
            data=np.transpose(np.asarray([dt1,dt2,dt3,dt4,dt5,dt6,dt7,dt8]))
            self.mlp = MLPClassifier(hidden_layer_sizes=(100,100,100,100))
            fit=False
            while not(fit):
                self.mlp.fit(data,t)
                fit=(np.count_nonzero(np.asarray((self.mlp.predict(data)==t)))==len(t))
            print(fit)
        else:
            self.mlp=md
    def check(self,ihu):
        D=np.abs(np.asarray(ihu))
        print(D)
        dt1=np.log10(D[0]*D[1])
        dt2=np.log10(D[2])
        dt3=np.log10(D[3])
        dt4=np.log10(D[4])
        dt5=np.log10(D[5])
        dt6=np.log10(D[6])
        dt7=(D[7])
        dt8=(D[8])
        data=np.transpose(np.asarray([dt1,dt2,dt3,dt4,dt5,dt6,dt7,dt8]))
        return self.mlp.predict([data])
