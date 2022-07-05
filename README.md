# Implementation-of-SVM-For-Spam-Mail-Detection

## AIM:
To write a program to implement the SVM For Spam Mail Detection.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Moodle-Code Runner

## Algorithm
1. Import the required packages.
2. Import the dataset to operate on.
3. Split the dataset.
4. Predict the required output.
5. End the program.
## Program:
```
/*
Program to implement the SVM For Spam Mail Detection..
Developed by: Kaarthikeyan.S
RegisterNumber: 212220040068 
*/
import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_csv('Spam.csv',encoding='latin-1')
df = data.drop(['Unnamed: 2','Unnamed: 3','Unnamed: 4'],axis=1)

df.head()

df.info()

df.isnull().sum()

x=df["v1"].values
y=df["v2"].values

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size = 0.2,random_state=0)

from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer()

x_train=cv.fit_transform(x_train)
x_test = cv.transform(x_test)

from sklearn.svm import SVC
svc = SVC()
svc.fit(x_train,y_train)

y_pred = svc.predict(x_test)
y_pred

from sklearn import metrics
accuracy = metrics.accuracy_score(y_test,y_pred)
accuracy

```

## Output:

<img width="479" alt="8ml 1" src="https://user-images.githubusercontent.com/94525701/173197044-22689695-61c6-476f-ba6d-b9ec60ae1e57.png">
<img width="479" alt="8ml2" src="https://user-images.githubusercontent.com/94525701/173197050-1483f6b6-ea09-4bce-a2e7-070dbd009387.png">
<img width="479" alt="8ml3" src="https://user-images.githubusercontent.com/94525701/173197055-f6db3f6f-358a-463c-a08b-f24691c4099d.png">
<img width="621" alt="8ml4" src="https://user-images.githubusercontent.com/94525701/173197061-07b11fb2-39cc-401e-8810-bfbd29c5e28c.png">
<img width="631" alt="8ml5" src="https://user-images.githubusercontent.com/94525701/173197066-21693412-1582-449e-bb73-f87971090b3d.png">


## Result:
Thus the program to implement the SVM For Spam Mail Detection is written and verified using python programming.
