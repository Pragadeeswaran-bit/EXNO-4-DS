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
Name : Pragadeeswaran L
Reg.No : 212223240120
```
```
import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
data=pd.read_csv("/content/income(1) (1).csv",na_values=[ " ?"])
data
```
![Screenshot 2024-09-30 152220](https://github.com/user-attachments/assets/2e942a96-88ea-486b-be6f-788efed41c2a)
```
data.isnull().sum()
```
![Screenshot 2024-09-30 152300](https://github.com/user-attachments/assets/12ec0bfc-9749-4c2a-a63e-2eaf6d2edb05)
```
missing=data[data.isnull().any(axis=1)]
missing
```
![Screenshot 2024-09-30 152438](https://github.com/user-attachments/assets/bbf06c2c-08c6-4b5d-9127-a984daee1bc9)
```
data2=data.dropna(axis=0)
data2
```
![Screenshot 2024-09-30 152522](https://github.com/user-attachments/assets/4d2c203b-a24e-4725-830b-da77a08958cf)
```
sal=data["SalStat"]
data2["SalStat"]=data["SalStat"].map({' less than or equal to 50,000':0,' greater than 50,000':1})
print(data2['SalStat'])
```
![Screenshot 2024-09-30 152625](https://github.com/user-attachments/assets/f622cc6c-1877-4d3e-a754-1ef9c5554517)

```
sal2=data2['SalStat']
dfs=pd.concat([sal,sal2],axis=1)
dfs
```
![Screenshot 2024-09-30 152828](https://github.com/user-attachments/assets/63ba6616-86ca-4a7d-ae9f-0c8a38d0d7fc)

```
data2
```
![Screenshot 2024-09-30 152904](https://github.com/user-attachments/assets/e0ed9374-58af-4d8c-836e-15f8308f5528)

```
new_data=pd.get_dummies(data2, drop_first=True)
new_data
```
![Screenshot 2024-09-30 153132](https://github.com/user-attachments/assets/14e4d22e-58db-40d2-a3c7-da2ca6ccf5a9)
```
columns_list=list(new_data.columns)
print(columns_list)
```
![Screenshot 2024-09-30 153522](https://github.com/user-attachments/assets/ad906b1d-1f0c-42f5-8631-d7db0928d1e0)
```
features=list(set(columns_list)-set(['SalStat']))
print(features)
```
![Screenshot 2024-09-30 155527](https://github.com/user-attachments/assets/40da0bcc-4c0a-45a9-b12b-126d687315a9)
```
y=new_data['SalStat'].values
print(y)
```
![Screenshot 2024-09-30 155550](https://github.com/user-attachments/assets/06c6e89a-6b8d-4725-a3e2-dbf47f8a79e8)
```
x=new_data[features].values
print(x)
```
![Screenshot 2024-09-30 155607](https://github.com/user-attachments/assets/c439e1b7-63d3-4d2d-9cac-fb8ce8e72cb2)
```
train_x,test_x,train_y,test_y=train_test_split(x,y,test_size=0.3,random_state=0)
KNN_classifier=KNeighborsClassifier(n_neighbors = 5)
KNN_classifier.fit(train_x,train_y)
```
![Screenshot 2024-09-30 155632](https://github.com/user-attachments/assets/b736cf8d-66b4-45cd-8763-17d25bb85401)
```
prediction=KNN_classifier.predict(test_x)
confusionMatrix=confusion_matrix(test_y, prediction)
print(confusionMatrix)
```
![Screenshot 2024-09-30 155659](https://github.com/user-attachments/assets/088d01e4-9ee2-4ff2-9818-ac6f88c29a9c)
```
accuracy_score=accuracy_score(test_y,prediction)
print(accuracy_score)
```
![Screenshot 2024-09-30 155720](https://github.com/user-attachments/assets/034a3dd8-9387-4fc0-a132-e5d5e4ad6f65)
```
print("Misclassified Samples : %d" % (test_y !=prediction).sum())
```
![Screenshot 2024-09-30 155737](https://github.com/user-attachments/assets/dbbdf1b4-5371-4d2c-a5b8-dbef1eff329b)
```
data.shape
```
![Screenshot 2024-09-30 155751](https://github.com/user-attachments/assets/f6988c41-d9bc-41ed-8d87-f475af6c5c4e)
```
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
![Screenshot 2024-09-30 155831](https://github.com/user-attachments/assets/ee948203-095f-4c9d-aac7-9e16b65b7bb3)
```
import pandas as pd
import numpy as np
from scipy.stats import chi2_contingency
import seaborn as sns
tips=sns.load_dataset('tips')
tips.head()
```
![Screenshot 2024-09-30 155854](https://github.com/user-attachments/assets/f4a557fa-0b87-4fc1-a53a-8b63ee793e54)
```
tips.time.unique()
```
![Screenshot 2024-09-30 155921](https://github.com/user-attachments/assets/793095fe-f80a-432c-9115-5f8d8e267386)
```
contingency_table=pd.crosstab(tips['sex'],tips['time'])
print(contingency_table)
```
![Screenshot 2024-09-30 155959](https://github.com/user-attachments/assets/06599ae1-ec29-4940-8ae2-4ba9bb110ca4)
```
chi2,p,_,_=chi2_contingency(contingency_table)
print(f"Chi-Square Statistics: {chi2}")
print(f"P-Value: {p}")
```
![Screenshot 2024-09-30 160016](https://github.com/user-attachments/assets/611d6bcc-c18c-440e-82b4-d7e651175c5a)

# RESULT:
Thus, Feature selection and Feature scaling has been used on thegiven dataset.
