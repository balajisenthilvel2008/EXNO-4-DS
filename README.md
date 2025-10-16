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
import pandas as pd
import numpy as np
from scipy import stats
df=pd.DataFrame(pd.read_csv(r"C:\Users\acer\Downloads\bmi.csv"))
df

```

![alt text](image.png)

```
df.head()
```

![alt text](image-1.png)

```
df.isnull().sum()
```
![alt text](image-2.png)

```
ss=StandardScaler()
df2=pd.DataFrame()
df2[columns]=ss.fit_transform(df1[columns])
df2.head()

```

![alt text](image-3.png)

```
norm=Normalizer()
df3=pd.DataFrame()
columns=['Height',"Weight","Index"]
df3=norm.fit_transform(df1[columns])
df3=pd.DataFrame(df3)
df3.columns=columns
df3.head()
```

![alt text](image-4.png)

```
mas=MaxAbsScaler()
df3=pd.DataFrame()
columns=['Height',"Weight","Index"]
df3=mas.fit_transform(df1[columns])
df3=pd.DataFrame(df3)
df3.columns=columns
df3.head()
```

![alt text](image-5.png)

```
rs=RobustScaler()
df3=pd.DataFrame()
columns=['Height',"Weight","Index"]
df3=rs.fit_transform(df1[columns])
df3=pd.DataFrame(df3)
df3.columns=columns
df3.head()
```

![alt text](image-6.png)

```
min=MinMaxScaler()
df3=pd.DataFrame()
columns=['Height',"Weight","Index"]
df3=min.fit_transform(df1[columns])
df3=pd.DataFrame(df3)
df3.columns=columns
df3.head()
```

![alt text](image-7.png)

```
import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.feature_selection import RFE
from sklearn.linear_model import RidgeCV, LassoCV, Ridge, Lasso
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import mutual_info_classif
from sklearn.feature_selection import mutual_info_regression
from sklearn.feature_selection import chi2
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectFromModel
df=pd.DataFrame(pd.read_csv(r'D:\data science\titanic_dataset.csv'))
df
```

![alt text](image-8.png)

```
df.drop(['PassengerId', 'Name', 'Ticket', 'Cabin'], axis=1, inplace=True)
df['Age'].fillna(df['Age'].median(), inplace=True)
df['Embarked'].fillna(df['Embarked'].mode()[0], inplace=True)
df = pd.get_dummies(df, columns=['Sex', 'Embarked'], drop_first=True)
X = df.drop('Survived', axis=1)
y = df['Survived']
rf = RandomForestClassifier(random_state=42)
rf.fit(X, y)
selector = SelectFromModel(rf, prefit=True)
X_selected = selector.transform(X)
selected_features = X.columns[selector.get_support()]
print("Selected Features:")
print(selected_features.tolist())
```

![alt text](image-9.png)

# RESULT:
Thus we have performed feature selection and feature scaling successfully.

