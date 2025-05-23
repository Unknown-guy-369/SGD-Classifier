# SGD-Classifier
## AIM:
To write a program to predict the type of species of the Iris flower using the SGD Classifier.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm

1. **Import Libraries and Load Dataset**  
   - Load the Iris dataset using `load_iris()` from `sklearn.datasets`.  
   - Convert it into a pandas DataFrame for easy handling.

2. **Preprocess the Data**  
   - Separate features (`x`) and target (`y`).  
   - Split the data into training and testing sets using `train_test_split()`.

3. **Train the Model**  
   - Initialize an `SGDClassifier` with a maximum of 1000 iterations.  
   - Train the model using `fit()` on the training data.

4. **Evaluate the Model**  
   - Predict the results using the test data.  
   - Calculate the accuracy using `accuracy_score()`.  
   - Display the confusion matrix and classification report.

## Program:
```
/*
Program to implement the prediction of iris species using SGD Classifier.
Developed by: Abishek Priyan M
RegisterNumber: 212224240004
*/
```

```py
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score,confusion_matrix,classification_report

iris = load_iris()

df  = pd.DataFrame(data=iris.data,columns=iris.feature_names)
df['target'] = iris.target

print(df.head())

x=df.drop('target',axis=1)
y=df['target']

x_train,x_test,y_train,y_test= train_test_split(x,y,test_size=0.2,random_state=42)

sgd_clf = SGDClassifier(max_iter=1000,tol=1e-3)

sgd_clf.fit(x_train,y_train)

y_pred = sgd_clf.predict(x_test)

accuracy = accuracy_score(y_test,y_pred)
print(f"Accuracy: {accuracy:.3f}")

cm = confusion_matrix(y_test,y_pred)
print("Confusion Matrix:")
print(cm)

classification_report1 = classification_report(y_test,y_pred)
print(classification_report1)

```

## Output:
![image](https://github.com/user-attachments/assets/60849142-ea0a-4274-9cc3-6f5c65873d35)

## Result:
Thus, the program to implement the prediction of the Iris species using SGD Classifier is written and verified using Python programming.
