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
C=np.logspace(-10,10, num=11)
P=np.zeros(len(C))


R=np.logspace(-10,1, num=11)
P=np.zeros(len(C)*len(R))
i=0
print ("Entering Training For Loop for Radial Basis Function Kernal")
for c in C:
	for r in R:
		clf = svm.SVC(C=c,kernel='rbf', gamma=r)
		clf.fit(Xt,yt)
		Pe = 1 - clf.score(Xh,yh)
		print("i= ", i, " r= ", r, " Pe= ", Pe, " C= ", c, "clf.n_support_", clf.n_support_)
		P[i]=Pe
		i=i+1
i=np.argmin(P)
C_opt=C[int(i/len(R))]
R_opt=R[i%len(R)]
clf = svm.SVC(C=C_opt,kernel='rbf', gamma=R_opt)
clf.fit(Xtrain,ytrain)
Pe=1-clf.score(Xtest,ytest)
sv = clf.support_vectors_
print ("For Radial Basis Function kernal Value of  optimal C is ", C_opt," and optimal r is ", R_opt," with Pe (for test dataset) ", Pe)
print ("Number of Support Vectors for each class is given by ", np.sum(clf.n_support_))


d = 1 - np.squeeze(np.sign(clf.dual_coef_))*clf.decision_function(sv)
d_sorted = np.argsort(d)
d_sorted = d_sorted[-16:]
SV_top=sv[d_sorted,:]
oid=clf.support_
y_sv=ytrain[oid[d_sorted]]
plt.figure(1)
f, axarr = plt.subplots(4, 4)   
for p in range(4):
    for q in range(4):
        axarr[p, q].imshow(SV_top[p*4+q].reshape((28,28)), cmap='gray') 
        axarr[p, q].set_title('{label}'.format(label=int(y_sv[p*4+q]))) 
        
#plt.show()

C=np.logspace(-10,10, num=21)
P=np.zeros(len(C))
i=0
print ("Entering Training For Loop for Linear Kernal")
for c in C:
	clf = svm.SVC(C=c,kernel='linear')
	clf.fit(Xt,yt)
	Pe = 1 - clf.score(Xh,yh)
	print("i= ", i, " Pe= ", Pe, " C= ", c, "clf.n_support_", clf.n_support_)
	P[i]=Pe
	i=i+1
i=np.argmin(P)
C_opt=C[i]
clf = svm.SVC(C=C_opt,kernel='linear')
clf.fit(Xtrain,ytrain)
Pe=1-clf.score(Xtest,ytest)
sv = clf.support_vectors_
print ("For Linear Kernel optimal Value of C is ", C_opt," with Pe (for test dataset) ", Pe)
print ("Number of Support Vectors for each class is given by ", np.sum(clf.n_support_))

i=0
print ("Entering Training For Loop for Polynomial Kernal Degree 1")
for c in C:
	clf = svm.SVC(C=c,kernel='poly', degree=1)
	clf.fit(Xt,yt)
	Pe = 1 - clf.score(Xh,yh)
	print("i= ", i, " Pe= ", Pe, " C= ", c, "clf.n_support_", clf.n_support_)
	P[i]=Pe
	i=i+1
i=np.argmin(P)
C_opt=C[i]
clf = svm.SVC(C=C_opt,kernel='poly', degree=1)
clf.fit(Xtrain,ytrain)
Pe=1-clf.score(Xtest,ytest)
sv = clf.support_vectors_
print ("For Polynomial Kernel of Degree 1 optimal Value of C is ", C_opt," with Pe (for Test dataset) ", Pe)
print ("Number of Support Vectors for each class is given by ", np.sum(clf.n_support_))

plt.figure(2)
d = 1 - np.squeeze(np.sign(clf.dual_coef_))*clf.decision_function(sv)
d_sorted = np.argsort(d)
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

P=np.zeros(len(C))
i=0
print ("Entering Training For Loop for Polynomial Kernal Degree 2")
for c in C:
	clf = svm.SVC(C=c,kernel='poly', degree=2)
	clf.fit(Xt,yt)
	Pe = 1 - clf.score(Xh,yh)
	print("i= ", i, " Pe= ", Pe, " C= ", c, "clf.n_support_", clf.n_support_)
	P[i]=Pe
	i=i+1
i=np.argmin(P)
C_opt=C[i]
clf = svm.SVC(C=C_opt,kernel='poly', degree=2)
clf.fit(Xtrain,ytrain)
Pe=1-clf.score(Xtest,ytest)
sv = clf.support_vectors_
print ("For Polynomial Kernel of Degree 2 optimal Value of C is ", C_opt," with Pe (for test dataset) ", Pe)
print ("Number of Support Vectors for each class is given by ", np.sum(clf.n_support_))

d = 1 - np.squeeze(np.sign(clf.dual_coef_))*clf.decision_function(sv)
d_sorted = np.argsort(d)
d_sorted = d_sorted[-16:]
SV_top=sv[d_sorted,:]
oid=clf.support_
y_sv=ytrain[oid[d_sorted]]

plt.figure(3)
f, axarr = plt.subplots(4, 4)   
for p in range(4):
    for q in range(4):
        axarr[p, q].imshow(SV_top[p*4+q].reshape((28,28)), cmap='gray') 
        axarr[p, q].set_title('{label}'.format(label=int(y_sv[p*4+q]))) 
        
plt.show()
