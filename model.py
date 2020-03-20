# Importing relevant libraries 

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle

# Loading data

df = pd.read_csv('ELS_02_StudentFile.csv',low_memory=False)
df.shape

# This ML model predicts college completion 
# based on a high schooler's level of financial motivation for studying
# Data source: Education Longitudinal Study of 2002 

# Target for logistic regression: received bachelor's degree
    # F3TZBACHLTDT - Date of first known bachelor's degree 
        
# Relevant inputs for logistic regression:
    
    # BYS83A - Mother's highest level of education
    # BYS89H - Studies to increase job opportunities
        # 1 - Almost never
        # 2 - Sometimes
        # 3 - Often
        # 4 - Almost always
    # BYS89P - Studies to ensure financial security 
        # 1 - Almost never
        # 2 - Sometimes
        # 3 - Often
        # 4 - Almost always
   
       
df = df.rename(columns={'STU_ID':'id','BYS83A':'mom_edu','BYS89H':'study_for_job',
                       'BYS89P':'study_for_security','F3TZBACHLTDT':'date_firstb'})

# Keeping features of interest

data_1 = df[['id','study_for_job','mom_edu','study_for_security','date_firstb']]  


# Dealing with missing/invalid data

data_1 = data_1[ (data_1['mom_edu']>=1) & (data_1['study_for_job']>=1)
               & (data_1['study_for_security']>=1) 
                & (data_1['date_firstb'] != -4) & (data_1['date_firstb'] != -9) ]
        
print("%d observations remaining" % (len(data_1)))

# Creating target: received bachelor's degree 

targets = np.where( (data_1['date_firstb']>=2007), 1, 0)

data_1['bachelors'] = targets

data_with_targets = data_1.drop(['date_firstb','id'],axis=1)
data_with_targets.reset_index()

# Modifying feature values 

data_with_targets['study_for_job'] = data_with_targets['study_for_job'].map({1:0, 2:0, 3:1, 4:1})
data_with_targets['study_for_security'] = data_with_targets['study_for_security'].map({1:0, 2:0, 3:1, 4:1})

# TRAIN/TEST SPLIT

from sklearn.model_selection import train_test_split

data_cleaned = data_with_targets.iloc[:,:-1]

# Split
x_train, x_test, y_train, y_test = train_test_split(data_cleaned,targets,train_size=0.9,random_state=77)

print("Train/Test Split Results:")
print("Train:")
print(len(x_train))
print("Test")
print(len(x_test))

# Logistic Regression with Sklearn

from sklearn.linear_model import LogisticRegression
from sklearn import metrics

# Training the model
reg = LogisticRegression(solver='liblinear')
reg.fit(x_train,y_train)

# Assessing Accuracy
print("Reg score: %f" % (reg.score(x_train,y_train)))

# Summary table
feature_name = data_cleaned.columns.values
summary_table = pd.DataFrame(columns=['Feature name'],data = feature_name)
summary_table['Coefficient'] = np.transpose(reg.coef_)


summary_table.index = summary_table.index + 1
summary_table.loc[0] = ['Intercept',reg.intercept_[0]]
summary_table = summary_table.sort_index()
summary_table

summary_table['Odds ratio'] = np.exp(summary_table.Coefficient)


# Testing model
print("Reg score: %f" % reg.score(x_test,y_test))

# Saving the model
pickle.dump(reg, open('model.pkl','wb'))

model = pickle.load(open('model.pkl','rb'))
