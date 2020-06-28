
from sklearn import datasets

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression,LogisticRegression
from sklearn.svm import SVC
import pickle

iris=datasets.load_iris()
#print(iris)
X=iris.data
y=iris.target

x_train,x_test,y_train,y_test=train_test_split(X,y)
lin_reg=LinearRegression()
log_reg=LogisticRegression()
svc_model=SVC()
lin_reg=lin_reg.fit(x_train,y_train)
log_reg=log_reg.fit(x_train,y_train)
svc_model=svc_model.fit(x_train,y_train)


pickle.dump(lin_reg,open('lin_model.pkl','wb'))
pickle.dump(log_reg,open('log_model.pkl','wb'))
pickle.dump(svc_model,open('svc_model.pkl','wb'))