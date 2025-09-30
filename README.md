## EXNO-3-DS

# AIM:
To read the given data and perform Feature Encoding and Transformation process and save the data to a file.

# ALGORITHM:
STEP 1:Read the given Data.
STEP 2:Clean the Data Set using Data Cleaning Process.
STEP 3:Apply Feature Encoding for the feature in the data set.
STEP 4:Apply Feature Transformation for the feature in the data set.
STEP 5:Save the data to the file.

# FEATURE ENCODING:
1. Ordinal Encoding
An ordinal encoding involves mapping each unique label to an integer value. This type of encoding is really only appropriate if there is a known relationship between the categories. This relationship does exist for some of the variables in our dataset, and ideally, this should be harnessed when preparing the data.
2. Label Encoding
Label encoding is a simple and straight forward approach. This converts each value in a categorical column into a numerical value. Each value in a categorical column is called Label.
3. Binary Encoding
Binary encoding converts a category into binary digits. Each binary digit creates one feature column. If there are n unique categories, then binary encoding results in the only log(base 2)ⁿ features.
4. One Hot Encoding
We use this categorical data encoding technique when the features are nominal(do not have any order). In one hot encoding, for each level of a categorical feature, we create a new variable. Each category is mapped with a binary variable containing either 0 or 1. Here, 0 represents the absence, and 1 represents the presence of that category.

# Methods Used for Data Transformation:
  # 1. FUNCTION TRANSFORMATION
• Log Transformation
• Reciprocal Transformation
• Square Root Transformation
• Square Transformation
  # 2. POWER TRANSFORMATION
• Boxcox method
• Yeojohnson method

# CODING AND OUTPUT:

```
import pandas as pd
df=pd.read_csv("Encoding Data.csv")
df

```
<img width="1121" height="552" alt="Screenshot 2025-09-30 104848" src="https://github.com/user-attachments/assets/170bf25f-5161-486b-9e71-94e0702d4d82" />

```
from sklearn.preprocessing import LabelEncoder,OrdinalEncoder
pm=['Hot','Warm','Cold']
e1=OrdinalEncoder(categories=[pm])
e1.fit_transform(df[["ord_2"]])
```

<img width="1025" height="370" alt="Screenshot 2025-09-30 104855" src="https://github.com/user-attachments/assets/71c5b6b4-31aa-4519-bbf9-2d9f91d60af7" />


```
df['bo2']=e1.fit_transform(df[["ord_2"]])
df

```

<img width="999" height="518" alt="Screenshot 2025-09-30 104901" src="https://github.com/user-attachments/assets/8bdeaded-67ea-49de-940f-4b54cd12e6e7" />

```
le=LabelEncoder()
dfc=df.copy()
dfc['ord_2']=le.fit_transform(dfc['ord_2'])
dfc
```


<img width="1183" height="575" alt="Screenshot 2025-09-30 104908" src="https://github.com/user-attachments/assets/8234dd73-7b20-4708-bd6b-2634b7f1e7f6" />

```
from sklearn.preprocessing import OneHotEncoder

ohe = OneHotEncoder(sparse_output=False)   
df2 = df.copy()

enc = pd.DataFrame(
    ohe.fit_transform(df2[["nom_0"]]),
    columns=ohe.get_feature_names_out(["nom_0"]),
    index=df2.index
)

df2 = pd.concat([df2, enc], axis=1)
df2
```

<img width="1073" height="791" alt="Screenshot 2025-09-30 104926" src="https://github.com/user-attachments/assets/da12ac3e-e168-41dc-8d06-5f1add9e4e3e" />

```
pd.get_dummies(df2,columns=["nom_0"])
```


<img width="1376" height="494" alt="Screenshot 2025-09-30 104936" src="https://github.com/user-attachments/assets/3bc50be6-0b45-448c-afa3-efeed67a2630" />

```
pip install --upgrade category_encoders

```

<img width="1667" height="482" alt="Screenshot 2025-09-30 104955" src="https://github.com/user-attachments/assets/50c3f8ea-b40e-480b-8e3f-68282778625c" />

```

from category_encoders import BinaryEncoder
df=pd.read_csv("data.csv")
df
```

<img width="1378" height="533" alt="Screenshot 2025-09-30 105004" src="https://github.com/user-attachments/assets/8ad613fa-a47e-4169-ada3-335b2df20dfe" />

```
be=BinaryEncoder()
nd=be.fit_transform(df['Ord_2'])
df
```

<img width="1106" height="555" alt="Screenshot 2025-09-30 105010" src="https://github.com/user-attachments/assets/ab21266c-515a-4cf2-93e6-b0ff97c98186" />


```

dfb=pd.concat([df,nd],axis=1)
dfb

```


<img width="1263" height="532" alt="Screenshot 2025-09-30 105016" src="https://github.com/user-attachments/assets/dd158b7e-5a2c-498c-869c-2fbff11e62a6" />

```
from category_encoders import TargetEncoder
te=TargetEncoder()
CC=df.copy()
new=te.fit_transform(X=CC["City"],y=CC["Target"])
CC=pd.concat([CC,new],axis=1)
CC
```

<img width="1162" height="609" alt="Screenshot 2025-09-30 105022" src="https://github.com/user-attachments/assets/6dede0bb-1a49-4567-bc16-5a7300646e56" />


```
import pandas as pd
from scipy import stats
import numpy as np
df=pd.read_csv("Data_to_Transform.csv")
df
```


<img width="1261" height="666" alt="Screenshot 2025-09-30 105028" src="https://github.com/user-attachments/assets/1a6dc803-8978-40c7-a416-2659c54ff189" />

```

df.skew()

```

<img width="1057" height="327" alt="Screenshot 2025-09-30 105033" src="https://github.com/user-attachments/assets/ae5859ac-db1b-4cb3-9e7e-3210165d4ae3" />

```

np.log(df["Highly Positive Skew"])

```

<img width="1041" height="624" alt="Screenshot 2025-09-30 105038" src="https://github.com/user-attachments/assets/4699c1b3-0b45-4b89-bc07-5d89e7f216ec" />

```

np.reciprocal(df["Moderate Positive Skew"])

```

<img width="980" height="628" alt="Screenshot 2025-09-30 105045" src="https://github.com/user-attachments/assets/11113dea-d414-45d2-9b46-32dc0391a145" />

```

np.sqrt(df["Highly Positive Skew"])

```

<img width="877" height="609" alt="Screenshot 2025-09-30 105050" src="https://github.com/user-attachments/assets/a15ed776-b9d4-44a7-98d7-9a0dee30c160" />

```

np.square(df["Highly Positive Skew"])

```

<img width="1086" height="617" alt="Screenshot 2025-09-30 105056" src="https://github.com/user-attachments/assets/f4452170-e318-4278-8810-350df90403f8" />

```

df["Highly Positive Skew_boxcox"], parameters=stats.boxcox(df["Highly Positive Skew"])
df

```


<img width="1552" height="601" alt="Screenshot 2025-09-30 105105" src="https://github.com/user-attachments/assets/fbc70e8b-ecf1-4ab6-a635-5a44a59cc61b" />

```

df.skew()

```

<img width="790" height="360" alt="Screenshot 2025-09-30 105110" src="https://github.com/user-attachments/assets/8747e841-0f69-453e-996d-d6d8c9fec74c" />


```


df["Highly Negative Skew_yeojohnson"],parameters=stats.yeojohnson(df["Highly Negative Skew"])
df.skew()


```

<img width="1166" height="412" alt="Screenshot 2025-09-30 105116" src="https://github.com/user-attachments/assets/f1c5a70d-98e0-4b35-b07a-3113eaab8398" />

```

from sklearn.preprocessing import QuantileTransformer
qt=QuantileTransformer(output_distribution='normal')
df["Moderate Negative Skew_1"]=qt.fit_transform(df[["Moderate Negative Skew"]])
df

```


<img width="1786" height="660" alt="Screenshot 2025-09-30 105125" src="https://github.com/user-attachments/assets/751b85b2-ea2a-4d88-918d-d82745f6ff70" />



```

import seaborn as sns
import statsmodels.api as sm
import matplotlib.pyplot as plt
sm.qqplot(df["Moderate Negative Skew"],line='45')
plt.show()

```


<img width="1012" height="699" alt="Screenshot 2025-09-30 105131" src="https://github.com/user-attachments/assets/97e78707-43e4-4323-a0fb-1916eb7d7dbf" />



```

sm.qqplot(np.reciprocal(df["Moderate Negative Skew"]),line='45')
plt.show()

```

<img width="1158" height="638" alt="Screenshot 2025-09-30 105139" src="https://github.com/user-attachments/assets/f9d3f6cd-95e3-4bbe-ba52-20ac37092cdd" />

```

from sklearn.preprocessing import QuantileTransformer
qt=QuantileTransformer(output_distribution='normal',n_quantiles=891)
df["Moderate Negative Skew"]=qt.fit_transform(df[["Moderate Negative Skew"]])
sm.qqplot(df["Moderate Negative Skew"],line='45')
plt.show()

```

<img width="1186" height="709" alt="Screenshot 2025-09-30 105147" src="https://github.com/user-attachments/assets/6d414a81-eda4-46a5-983b-6b1f7a4a680d" />

```


df["Highly Negative Skew_1"]=qt.fit_transform(df[["Highly Negative Skew"]])
sm.qqplot(df["Highly Negative Skew"],line='45')
plt.show()


```

<img width="1125" height="666" alt="Screenshot 2025-09-30 105152" src="https://github.com/user-attachments/assets/8b43bb01-f3f3-41a5-868f-d7307a663eb5" />



```

dt=pd.read_csv("titanic_dataset.csv")
dt


```

<img width="1551" height="575" alt="Screenshot 2025-09-30 105210" src="https://github.com/user-attachments/assets/0762473f-d361-46c6-9b5b-8bc7ffda8d0c" />

```

from sklearn.preprocessing import QuantileTransformer
qt=QuantileTransformer(output_distribution='normal',n_quantiles=891)
dt["Age_1"]=qt.fit_transform(dt[["Age"]])
sm.qqplot(dt['Age'],line='45') 
plt.show()

```

<img width="1432" height="708" alt="Screenshot 2025-09-30 105218" src="https://github.com/user-attachments/assets/13fc11f6-5f0e-4c4c-a007-d9df539ba6cb" />


```

sm.qqplot(df["Highly Negative Skew_1"],line='45')
plt.show()

```

<img width="1126" height="641" alt="Screenshot 2025-09-30 105227" src="https://github.com/user-attachments/assets/7980abc5-5b43-4490-b6d6-64f4215aeec1" />



























# RESULT:
 Thus the given data, Feature Encoding, Transformation process and save the data to a file
was performed successfully.

       
