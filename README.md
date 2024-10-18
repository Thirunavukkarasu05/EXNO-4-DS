# EXNO:4-DS
NAME : THIRUNAVUKKARASU P
REG NO:212222040173

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
import pandas as pd
from scipy import stats
import numpy as np
df=pd.read_csv("/content/bmi.csv")
df.head()
```
<img width="216" alt="image" src="https://github.com/KayyuruTharani/EXNO-4-DS/assets/142209319/2aea6717-ef53-42bb-b94c-5228a301c51b">

```
df.dropna()
```

<img width="214" alt="image" src="https://github.com/KayyuruTharani/EXNO-4-DS/assets/142209319/8b146d52-1bff-4afe-95ad-1fd1b34c1890">

```
max_vals=np.max(np.abs(df[['Height','Weight']]))
max_vals

```
<img width="46" alt="image" src="https://github.com/KayyuruTharani/EXNO-4-DS/assets/142209319/76a9d44b-b7de-43b0-b4c5-4fa2369743cd">

```
from sklearn.preprocessing import StandardScaler
sc=StandardScaler()
df[['Height','Weight']]=sc.fit_transform(df[['Height','Weight']])
df.head(10)
```

<img width="204" alt="image" src="https://github.com/KayyuruTharani/EXNO-4-DS/assets/142209319/c1e1b5ab-3592-4f38-9696-9e394eb38a08">

```
from sklearn.preprocessing import MinMaxScaler
scaler=MinMaxScaler()
df[['Height','Weight']]=scaler.fit_transform(df[['Height','Weight']])
df.head(10)
```

<img width="196" alt="image" src="https://github.com/KayyuruTharani/EXNO-4-DS/assets/142209319/cc1b1e0b-0a00-455f-b3aa-2820d2eaea4c">

```
from sklearn.preprocessing import Normalizer
scaler=Normalizer()
df[['Height','Weight']]=scaler.fit_transform(df[['Height','Weight']])
df
```

<img width="214" alt="image" src="https://github.com/KayyuruTharani/EXNO-4-DS/assets/142209319/d55c9d08-42a1-4106-9fb0-dad172a28cfd">

```
df1=pd.read_csv("/content/bmi.csv")
from sklearn.preprocessing import MaxAbsScaler
scaler=MaxAbsScaler()
df1[['Height','Weight']]=scaler.fit_transform(df1[['Height','Weight']])
df1

```
<img width="210" alt="image" src="https://github.com/KayyuruTharani/EXNO-4-DS/assets/142209319/1a21a595-3a30-43de-9dd9-abc59ff86dd8">

```
df2=pd.read_csv("/content/bmi.csv")
from sklearn.preprocessing import RobustScaler
scaler=RobustScaler()
df2[['Height','Weight']]=scaler.fit_transform(df2[['Height','Weight']])
df2.head()
```

<img width="203" alt="image" src="https://github.com/KayyuruTharani/EXNO-4-DS/assets/142209319/df58bc4e-e7c9-49d4-ba54-c61123297ee1">

```

import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score,confusion_matrix
data=pd.read_csv('/content/income(1) (1).csv',na_values=[" ?"])
data
```

<img width="818" alt="image" src="https://github.com/KayyuruTharani/EXNO-4-DS/assets/142209319/1eb413c2-b5a1-4d89-a654-d96a5cbc64b0">

```
data.isnull().sum()
```

<img width="119" alt="image" src="https://github.com/KayyuruTharani/EXNO-4-DS/assets/142209319/9a0b6c2c-4c0c-46b9-91ad-475f04ce3562">

```
missing=data[data.isnull().any(axis=1)]
missing
```

<img width="794" alt="image" src="https://github.com/KayyuruTharani/EXNO-4-DS/assets/142209319/2b5bbfb3-9315-472c-a097-8adee8f4578f">

```
data2 = data.dropna(axis=0)
data2

```
<img width="812" alt="image" src="https://github.com/KayyuruTharani/EXNO-4-DS/assets/142209319/e0209e9d-dfa7-4245-8d6a-6c695e6728bc">

```
sal=data['SalStat']
data2['SalStat']=data2['SalStat'].map({' less than or equal to 50,000':0,' greater than 50,000':1})
print(data2['SalStat'])

```
<img width="670" alt="image" src="https://github.com/KayyuruTharani/EXNO-4-DS/assets/142209319/c3919808-6ed9-432e-b83e-9aff94a35994">

```

sal2=data2['SalStat']
dfs=pd.concat([sal,sal2],axis=1)
dfs
```

<img width="204" alt="image" src="https://github.com/KayyuruTharani/EXNO-4-DS/assets/142209319/a2ce5564-7e32-481d-9547-e5943afb5c15">

```
data2
```

<img width="745" alt="image" src="https://github.com/KayyuruTharani/EXNO-4-DS/assets/142209319/a4fc1de3-8051-4743-81c6-c054bcb95f7d">

```
new_data=pd.get_dummies(data2, drop_first=True)
new_data
```

<img width="752" alt="image" src="https://github.com/KayyuruTharani/EXNO-4-DS/assets/142209319/7eef042e-6822-4ac5-a574-b19476d68a9a">

```
columns_list=list(new_data.columns)
print(columns_list)

```
<img width="901" alt="image" src="https://github.com/KayyuruTharani/EXNO-4-DS/assets/142209319/2958db03-4ee4-4d69-8a09-2154b7756727">

```
features=list(set(columns_list)-set(['SalStat']))
print(features)

```
<img width="892" alt="image" src="https://github.com/KayyuruTharani/EXNO-4-DS/assets/142209319/fd4a58e0-39ef-426f-89d3-ff7af0fffd98">

 ```
y=new_data['SalStat'].values
print(y)
```
[0 0 1 ... 0 0 0]

```
x = new_data[features].values
print(x)
```

<img width="203" alt="image" src="https://github.com/KayyuruTharani/EXNO-4-DS/assets/142209319/6ad70fd2-6946-4995-be0e-801b3b2f0212">

```
train_x,test_x,train_y,test_y = train_test_split(x,y,test_size=0.3, random_state=0)
KNN_classifier=KNeighborsClassifier(n_neighbors = 5)
KNN_classifier.fit(train_x,train_y)
```

<img width="140" alt="image" src="https://github.com/KayyuruTharani/EXNO-4-DS/assets/142209319/4c854a00-85d9-4d76-b57e-292d13ba84ee">

```
prediction = KNN_classifier.predict(test_x)
confusionMmatrix = confusion_matrix(test_y, prediction)
print(confusionMmatrix)

```
<img width="55" alt="image" src="https://github.com/KayyuruTharani/EXNO-4-DS/assets/142209319/2aae088f-279f-4f33-8137-b712dc580d9e">

```
accuracy_score=accuracy_score(test_y, prediction)
print(accuracy_score)

```
0.8392087523483258
```

print('Misclassified samples: %d' % (test_y != prediction).sum())
```

Misclassified samples: 1455

```
data.shape

(31978, 13)
```

## FEATURE SELECTION TECHNIQUES

```
import pandas as pd
import numpy as np
from scipy.stats import chi2_contingency
import seaborn as sns
tips=sns.load_dataset('tips')
tips.head()

```
<img width="257" alt="image" src="https://github.com/KayyuruTharani/EXNO-4-DS/assets/142209319/ee1738f1-e407-493b-a162-dc1ef444d0e4">

```
contingency_table=pd.crosstab(tips['sex'],tips['time'])
print(contingency_table)

```
<img width="139" alt="image" src="https://github.com/KayyuruTharani/EXNO-4-DS/assets/142209319/51bcc338-6dad-4de6-92c9-bfbbcfaafeb0">

```
chi2, p, _, _ = chi2_contingency(contingency_table)
print(f"Chi-Square Statistic: {chi2}")
print(f"P-value: {p}")

```
<img width="206" alt="image" src="https://github.com/KayyuruTharani/EXNO-4-DS/assets/142209319/72746275-4a20-4305-9e80-3bbf1ed0a806">

```
import pandas as pd
from sklearn.feature_selection import SelectKBest, mutual_info_classif, f_classif
data={
    'Feature1':[1,2,3,4,5],
    'Feature2': ['A','B','C','A','B'],
    'Feature3':[0,1,1,0,1],
    'Target' :[0,1,1,0,1]
}
df=pd.DataFrame(data)
X=df[['Feature1','Feature3']]
y=df['Target']
selector=SelectKBest(score_func=mutual_info_classif, k=1)
X_new = selector.fit_transform (X,y)
selected_feature_indices = selector.get_support(indices=True)
selected_features = X.columns[selected_feature_indices]
print("Selected Features:")
print(selected_features)

```
<img width="194" alt="image" src="https://github.com/KayyuruTharani/EXNO-4-DS/assets/142209319/66eac5dc-973c-4cee-833a-21a3ceb5fc61">



## RESULT:
To read the given data and perform Feature Scaling and Feature Selection process and save the data to a file is successful.
