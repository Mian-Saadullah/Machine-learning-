import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB, MultinomialNB, BernoulliNB

data={
    'hours':[1,2,3,4,5,6,7,8,9],
    'Marks':[20,30,35,40,50,65,75,85,90],
    'Result':['fail','fail','fail','fail','Pass','Pass','Pass','Pass','Pass']
}

dataset=pd.DataFrame(data)

print(dataset)

x=dataset[['hours','Marks']]
y=dataset['Result']

X_train,X_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=42)

model=GaussianNB()
model.fit(X_train,y_train)

prediction=model.predict(X_test)

print("predict data",prediction)
print("actual data",y_test.values)

# test the model

new_data = [
    [2, 25],
    [8, 88]
]

custom_prediction = model.predict(new_data)
print("Custom Predictions:", custom_prediction)