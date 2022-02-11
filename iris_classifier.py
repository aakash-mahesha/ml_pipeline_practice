# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import os

df=pd.read_csv('iris.data')

X=df.iloc[:,:-1]
y=df.iloc[:,-1]

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.3,stratify=y,random_state=7)

X_train.shape,X_test.shape,y_train.shape,y_test.shape



lb_encoder=LabelEncoder()

y_train_encoded=lb_encoder.fit_transform(y_train)
y_test_encoded=lb_encoder.transform(y_test)


rf=RandomForestClassifier()
rf.fit(X_train,y_train_encoded)
predictions=rf.predict(X_test)


from sklearn.metrics import classification_report
print(classification_report(y_test_encoded,predictions))


prediction_values=lb_encoder.inverse_transform(predictions)

result_df=pd.DataFrame(y_test.values.reshape(-1,1),columns=['Actual_values'])

result_df['Predicted Values']=prediction_values

os.mkdir(os.path.join(os.getcwd(),'results_dir'))
result_df.to_csv(os.path.join(os.getcwd(),'results_dir','resutls.csv'),index=False)
