# Implementation-of-Logistic-Regression-Using-Gradient-Descent

## AIM:
To write a program to implement the the Logistic Regression Using Gradient Descent.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm1. Import the required libraries.
2. Load the dataset.
3. Define X and Y array.
4. Define a function for costFunction,cost and gradient.
5. Define a function to plot the decision boundary. 6.Define a function to predict the 
   Regression value.

## Program:
```
/*
Program to implement the the Logistic Regression Using Gradient Descent.
Developed by: vishal 
RegisterNumber:  2122234184
*/
```
```
import numpy as np
import matplotlib.pyplot as plt
from scipy import optimize

data=np.loadtxt("ex2data1.txt",delimiter=',')
X=data[:,[0,1]]
y=data[:,2]

X[:5]

y[:5]

plt.figure()
plt.scatter(X[y==1][:,0],X[y==1][:,1],label="Admitted")
plt.scatter(X[y==0][:,0],X[y==0][:,1],label="Not Admitted")
plt.xlabel("Exam 1 score")
plt.ylabel("Exam 2 score")
plt.legend()
plt.show()

def sigmoid(z):
    return 1/(1+np.exp(-z))

plt.plot()
X_plot=np.linspace(-10,10,100)
plt.plot(X_plot,sigmoid(X_plot))
plt.show()

def costFunction (theta,X,y):
    h=sigmoid(np.dot(X,theta))
    J=-(np.dot(y,np.log(h))+np.dot(1-y,np.log(1-h)))/X.shape[0]
    grad=np.dot(X.T,h-y)/X.shape[0]
    return J,grad

X_train=np.hstack((np.ones((X.shape[0],1)),X))
theta=np.array([0,0,0])
J,grad=costFunction(theta,X_train,y)
print(J)
print(grad)

X_train=np.hstack((np.ones((X.shape[0],1)),X))
theta=np.array([-24,0.2,0.2])
J,grad=costFunction(theta,X_train,y)
print(J)
print(grad)

def cost (theta,X,y):
    h=sigmoid(np.dot(X,theta))
    J=-(np.dot(y,np.log(h))+np.dot(1-y,np.log(1-h)))/X.shape[0]
    return J

def gradient (theta,X,y):
    h=sigmoid(np.dot(X,theta))
    grad=np.dot(X.T,h-y)/X.shape[0]
    return grad

X_train=np.hstack((np.ones((X.shape[0],1)),X))
theta=np.array([0,0,0])
res=optimize.minimize(fun=cost,x0=theta,args=(X_train,y),method='Newton-CG',jac=gradient)
print(res.fun)
print(res.x)

def plotDecisionBoundary(theta,X,y):
    x_min,x_max=X[:,0].min()-1,X[:,0].max()+1
    y_min,y_max=X[:,1].min()-1,X[:,1].max()+1
    xx,yy=np.meshgrid(np.arange(x_min,x_max,0.1),np.arange(y_min,y_max,0.1))
    X_plot=np.c_[xx.ravel(),yy.ravel()]
    X_plot=np.hstack((np.ones((X_plot.shape[0],1)),X_plot))
    y_plot=np.dot(X_plot,theta).reshape(xx.shape)
    
    plt.figure()
    plt.scatter(X[y==1][:,0],X[y==1][:,1],label="Admitted")
    plt.scatter(X[y==0][:,0],X[y==0][:,1],label="Not Admitted")
    plt.contour(xx,yy,y_plot,levels=[0])
    plt.xlabel("Exam 1 score")
    plt.ylabel("Exam 2 score")
    plt.legend()
    plt.show()


plotDecisionBoundary(res.x,X,y)

prob=sigmoid(np.dot(np.array([1,45,85]),res.x))
print(prob)

def predict(theta,X):
    X_train =np.hstack((np.ones((X.shape[0],1)),X))
    prob=sigmoid(np.dot(X_train,theta))
    return (prob>=0.5).astype(int)
np.mean(predict(res.x,X)==y)
```

## Output:

Array of x

![image](https://github.com/23013753/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/145634121/adba9bb9-8ee7-4521-865f-6541556e98ef)

Array of y

![image](https://github.com/23013753/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/145634121/c139e7cf-d762-4166-86bd-41e30276be31)

Exam 1 :Score graph

![image](https://github.com/23013753/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/145634121/a8bb8788-e304-4b17-a64f-5e419af27a43)


Sigmoid function graph

![image](https://github.com/23013753/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/145634121/7e30dc25-1f43-4009-adda-b125bdc7bf43)


X_train_grad value

![image](https://github.com/23013753/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/145634121/0488c1ba-5dd5-4051-a740-a93a1c75e795)


Y_train_grad value

![image](https://github.com/23013753/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/145634121/45d08099-e177-47d2-941a-3654cfbb76bf)


Print resx

![image](https://github.com/23013753/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/145634121/de9a4fb6-4f29-4683-83cd-44c8f373090d)


Decision boundary - graph for exam score

![image](https://github.com/23013753/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/145634121/746c6c23-aced-4649-aeaf-1b217de89468)


Proability value

![image](https://github.com/23013753/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/145634121/643c5118-4c95-4595-a72f-eb7eef062130)


Prediction value of mean

![image](https://github.com/23013753/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/145634121/b2724af8-10df-4290-b81f-392cc44f71d2)




## Result:
Thus the program to implement the the Logistic Regression Using Gradient Descent is written and verified using python programming.

