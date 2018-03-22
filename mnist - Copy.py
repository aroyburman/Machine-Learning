from sklearn.datasets import fetch_mldata
mnist = fetch_mldata('MNIST original')
import matplotlib.pyplot as plt
from sklearn import svm
import numpy as np

X = mnist.data
y = mnist.target
X4 = X[y==4,:]
X9 = X[y==9,:]
y4= y[y==4,]
y9= y[y==9,]
Xtrain=np.concatenate((X4[0:4000,],X9[0:4000,]), axis=0)
ytrain=np.concatenate((y4[0:4000,],y9[0:4000,]), axis=0)
Xt=np.concatenate((X4[0:2000,],X9[0:2000,]), axis=0)
yt=np.concatenate((y4[0:2000,],y9[0:2000,]), axis=0)
Xh=np.concatenate((X4[2000:4000,],X9[2000:4000,]), axis=0)
yh=np.concatenate((y4[2000:4000,],y9[2000:4000,]), axis=0)
Xtest=np.concatenate((X4[4000:,],X9[4000:,]), axis=0)
ytest=np.concatenate((y4[4000:,],y9[4000:,]), axis=0)
C=np.logspace(-10,5, num=10)
P=np.zeros(len(C))
m=2
n=3
print ("m/n= ", int(m/n))

P=np.zeros(len(C))
i=0
print ("Entering Training For Loop for Polynomial Kernal Degree 1")

clf = svm.SVC(C=0.000464,kernel='poly', degree=1)
clf.fit(Xt,yt)
Pe = 1 - clf.score(Xh,yh)
print("i= ", i, " Pe= ", Pe, " C= ", c, "clf.n_support_", clf.n_support_)
P[i]=Pe

i=np.argmin(P)
C_opt=C[i]
clf = svm.SVC(C=C_opt,kernel='poly', degree=1)
clf.fit(Xtrain,ytrain)
Pe=1-clf.score(Xtest,ytest)
sv = clf.support_vectors_
print ("For Polynomial Kernel of Degree 1 optimal Value of C is ", C_opt," with Pe (for Test dataset) ", Pe)
print ("Number of Support Vectors for each class is given by ", clf.n_support_)

plt.figure(2)
d = clf.decision_function(sv)
d_sorted = np.argsort(np.abs(d))
d_sorted = d_sorted[-16:]
SV_top=sv[d_sorted,:]
oid=clf.support_
y_sv=ytrain[oid[d_sorted]]

f, axarr = plt.subplots(4, 4)   
for p in range(4):
    for q in range(4):
        axarr[p, q].imshow(SV_top[p*4+q].reshape((28,28)), cmap='gray') 
        axarr[p, q].set_title('{label}'.format(label=int(y_sv[p*4+q]))) 
        
#plt.show()
