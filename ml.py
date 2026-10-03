import pandas as pd
from sklearn.model_selection import train_test_split 
from sklearn.linear_model import LinearRegression

data={
        'hours':[1,2,3,4,5,6,7,8],
        'marks':[20,35,40,50,60,70,80,95]
}
dataset=pd.DataFrame(data)
print(dataset)

X=dataset[['hours']]
y=dataset['marks']

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=45)

model=LinearRegression()
model.fit(X_train,y_train)

prediction=model.predict(X_test)

print("predict data",prediction)
print("actual data",y_test.values)