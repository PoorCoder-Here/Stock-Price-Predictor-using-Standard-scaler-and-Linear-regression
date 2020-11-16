from sklearn import datasets
from matplotlib import pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
import csv
from sklearn.metrics import accuracy_score
import numpy as np

f=open('#Enter location of file','w',newline='')
wrt=csv.writer(f,delimiter=',')
wrt.writerow(['Date_time ','Price '])
d=pd.read_csv('#Enter file location')
da=np.array(d['DATE '])
mo=np.array(d['MONTH '])
y=np.array(d['YEAR '])
h=np.array(d['HOUR '])
m=np.array(d['MINUTE '])
s=np.array(d['SECOND '])
p=np.array(d['PRICE '])
for i in range(0,len(da)):
    t=f'{h[i]}:{m[i]}:{s[i]}'
    price=p[i]
    wrt.writerow([t,price])
f.close()
da=pd.read_csv('#Enter location file')
ac=pd.to_timedelta(da['Date_time '])
print(ac)
da['Date_time ']=np.array(ac)
a=np.array(da['Date_time '])
b=da['Price ']
print(type(a[0]))
plt.plot(a,b)
plt.xticks(rotation=90)
plt.show()
da=pd.read_csv('#Enter file location')
da['Date_time ']=pd.to_datetime(da['Date_time '])
a=da[['Date_time ']]
b=da[['Price ']]
sc=StandardScaler()
xtrain,xtest,ytrain,ytest=train_test_split(a,b,test_size=0.3,random_state=1)
sc.fit(xtrain)
xtrain_std=sc.transform(xtrain)
xtest_std=sc.transform(xtest)
model=LinearRegression(n_jobs=500,normalize=True)
model.fit(xtrain_std,ytrain)
pre=model.predict(xtest_std)
print(pre)
for i in range(0,len(pre)):
    print(pre[i][0],ytest[i][0])
print(model.score(xtest_std,ytest))
sc=StandardScaler()
airtel=pd.read_csv('#Enter location of file')
X=airtel[['DATE ','MONTH ','YEAR ','HOUR ','MINUTE ','SECOND ']]
Y=np.array(airtel[['PRICE ']])
xtrain,xtest,ytrain,ytest=train_test_split(X,Y,test_size=0.3,random_state=1)
sc.fit(xtrain)
xtrain_std=sc.transform(xtrain)
xtest_std=sc.transform(xtest)
model=LinearRegression(n_jobs=500,normalize=True)
model.fit(xtrain_std,ytrain)
pre=model.predict(xtest_std)
#print(pre[4][0],ytest[4][0])
for i in range(0,len(pre)):
    print(pre[i][0],ytest[i][0])
print(model.score(xtest_std,ytest))
x=i.data[:,[2,3]]
y=i.target
print(x)
print(y)
xtrain,xtest,ytrain,ytest=train_test_split(x, y, test_size=0.3,random_state=1,stratify=y)
sc.fit(xtrain)
xtrain_std=sc.transform(xtrain)
xtest_std=sc.transform(xtest)
model=LogisticRegression(random_state=1, n_jobs=50)
model.fit(xtrain_std,ytrain)
pre=model.predict(xtest_std)
print(ytest)
print(pre)
print(accuracy_score(ytest,pre))
