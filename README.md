# EXNO:4-DS
# AIM:
To read the given data and perform Feature Scaling and Feature Selection process and save the
data to a file.

# ALGORITHM:
STEP 1:Read the given Data.
STEP 2:Clean the Data Set using Data Cleaning Process.
STEP 3:Apply Feature Scaling for the feature in the data set.
STEP 4:Apply Feature Selection for the feature in the data set.
STEP 5:Save the data to the file.

# FEATURE SCALING:
1. Standard Scaler: It is also called Z-score normalization. It calculates the z-score of each value and replaces the value with the calculated Z-score. The features are then rescaled with x̄ =0 and σ=1
2. MinMaxScaler: It is also referred to as Normalization. The features are scaled between 0 and 1. Here, the mean value remains same as in Standardization, that is,0.
3. Maximum absolute scaling: Maximum absolute scaling scales the data to its maximum value; that is,it divides every observation by the maximum value of the variable.The result of the preceding transformation is a distribution in which the values vary approximately within the range of -1 to 1.
4. RobustScaler: RobustScaler transforms the feature vector by subtracting the median and then dividing by the interquartile range (75% value — 25% value).

# FEATURE SELECTION:
Feature selection is to find the best set of features that allows one to build useful models. Selecting the best features helps the model to perform well.
The feature selection techniques used are:
1.Filter Method
2.Wrapper Method
3.Embedded Method

# CODING AND OUTPUT:
      
```
NAME : THIRUNAVUKKARASU P
REG NO : 212222040173
```
```python
import pandas as pd
import numpy as np
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix

data=pd.read_csv("/content/income(1) (1).csv",na_values=[ " ?"])
data
```
![Screenshot 2024-10-03 105005](https://github.com/user-attachments/assets/0d9200b4-6128-4b3b-98e3-170372362ef5)

```python

data.isnull().sum()
```
![Screenshot 2024-10-03 105135](https://github.com/user-attachments/assets/b05b9502-8a24-4bfb-aa06-13ef1ae5f64f)

```python

missing=data[data.isnull().any(axis=1)]
missing
```
![image](https://github.com/user-attachments/assets/e10b79af-1d96-40f8-a8a1-5146569fcbc5)

```python

data2=data.dropna(axis=0)
data2
```
![image](https://github.com/user-attachments/assets/8c5b6868-7bcb-49aa-a739-11123eab2c47)

```python
sal=data["SalStat"]

data2["SalStat"]=data["SalStat"].map({' less than or equal to 50,000':0,' greater than 50,000':1})
print(data2['SalStat'])
```
![image](https://github.com/user-attachments/assets/38b17801-5354-492b-b271-48c6cfa672c6)

```python
sal2=data2['SalStat']

dfs=pd.concat([sal,sal2],axis=1)
dfs
```
![image](https://github.com/user-attachments/assets/dbb67bf2-8038-40d6-8ca6-af150e17bb3c)
```python
data2
```
![image](https://github.com/user-attachments/assets/f230751f-8c5e-4cad-8e59-282ff7ce9f59)
```python
new_data=pd.get_dummies(data2, drop_first=True)
new_data
```
![image](https://github.com/Yamunaasri/EXNO-4-DS/assets/115707860/f21819e3-a5bd-47e6-b1b7-9bc08b64bed9)
```python

columns_list=list(new_data.columns)
print(columns_list)
```
![image](https://github.com/Yamunaasri/EXNO-4-DS/assets/115707860/8af6f5ce-4d99-4ed6-9371-730aeaa5a56b)
```python


features=list(set(columns_list)-set(['SalStat']))
print(features)
```
![image](https://github.com/Yamunaasri/EXNO-4-DS/assets/115707860/5f31a677-7d30-417a-8044-d5db741cafbf)
```python
y=new_data['SalStat'].values
print(y)
```
![image](https://github.com/Yamunaasri/EXNO-4-DS/assets/115707860/f4c779af-4c87-449e-9daa-be5d8d275212)
```python

x=new_data[features].values
print(x)
```
![image](https://github.com/Yamunaasri/EXNO-4-DS/assets/115707860/4154db03-4c87-4b98-a13b-964f19bee9b0)
```python

train_x,test_x,train_y,test_y=train_test_split(x,y,test_size=0.3,random_state=0)

KNN_classifier=KNeighborsClassifier(n_neighbors = 5)

KNN_classifier.fit(train_x,train_y)
```
![image](https://github.com/Yamunaasri/EXNO-4-DS/assets/115707860/e5e02520-eb39-436c-ac2e-e43048c1d672)
```python

prediction=KNN_classifier.predict(test_x)

confusionMatrix=confusion_matrix(test_y, prediction)
print(confusionMatrix)
```
![image](https://github.com/Yamunaasri/EXNO-4-DS/assets/115707860/a6eedfe3-aedd-4500-958f-6faafd54f464)
```python

accuracy_score=accuracy_score(test_y,prediction)
print(accuracy_score)
```
![image](https://github.com/Yamunaasri/EXNO-4-DS/assets/115707860/0e56ff41-2f35-4d01-b479-53547391567b)
```python

print("Misclassified Samples : %d" % (test_y !=prediction).sum())
```
![image](https://github.com/Yamunaasri/EXNO-4-DS/assets/115707860/4af5ed3f-362a-40c6-a438-c89f31584e51)
```python

data.shape
```
![image](https://github.com/Yamunaasri/EXNO-4-DS/assets/115707860/1986f990-26e6-4b42-acfc-b2a6e52f8042)
```python

import pandas as pd
from sklearn.feature_selection import SelectKBest, mutual_info_classif, f_classif
data={
    'Feature1': [1,2,3,4,5],
    'Feature2': ['A','B','C','A','B'],
    'Feature3': [0,1,1,0,1],
    'Target'  : [0,1,1,0,1]
}

df=pd.DataFrame(data)
x=df[['Feature1','Feature3']]
y=df[['Target']]

selector=SelectKBest(score_func=mutual_info_classif,k=1)
x_new=selector.fit_transform(x,y)

selected_feature_indices=selector.get_support(indices=True)

selected_features=x.columns[selected_feature_indices]
print("Selected Features:")
print(selected_features)
```
![image](https://github.com/Yamunaasri/EXNO-4-DS/assets/115707860/20777b0d-3cdb-4ae9-80e4-1f76ed093191)
```python

import pandas as pd
import numpy as np
from scipy.stats import chi2_contingency

import seaborn as sns
tips=sns.load_dataset('tips')
tips.head()
```
![image](https://github.com/Yamunaasri/EXNO-4-DS/assets/115707860/6d6f7ff2-b1da-4568-9cd1-cb6fa9553cd6)
```python

tips.time.unique()
```
![image](https://github.com/Yamunaasri/EXNO-4-DS/assets/115707860/f77bc757-8a31-4a5d-be15-5a447e6549c6)
```python

contingency_table=pd.crosstab(tips['sex'],tips['time'])
print(contingency_table)
```
![image](https://github.com/Yamunaasri/EXNO-4-DS/assets/115707860/06365e9f-f51b-4cf6-ab04-8a136726a025)
```python

chi2,p,_,_=chi2_contingency(contingency_table)
print(f"Chi-Square Statistics: {chi2}")
print(f"P-Value: {p}")
```
![image](https://github.com/Yamunaasri/EXNO-4-DS/assets/115707860/6adc4da7-421c-458f-9ec6-f6158aa6f731)
# RESULT:
Thus, Feature selection and Feature scaling has been used on thegiven dataset.
