#For heart disease prediction
# import numpy as np
# import pandas as pd
# df = pd.read_csv('Heart_Disease_Prediction.csv')
# from sklearn.model_selection import train_test_split
# x = np.array(df.drop(columns='Heart Disease'))
# y = np.array(df['Heart Disease'])
# from sklearn.preprocessing import StandardScaler
# scaler = StandardScaler()
# scaler.fit(x)
# x_scaled = scaler.transform(x)
# x_train,x_test,y_train,y_test = train_test_split(x_scaled,y,train_size=0.8)
# from sklearn.linear_model import LogisticRegression
# clf = LogisticRegression()
# clf.fit(x_train,y_train)
# y_pred = clf.predict(x_test)
# print(y_pred)
# import joblib
# joblib.dump(clf, 'logistic_regression_model.joblib')


# For diabetes prediction
# import numpy as np
# import pandas as pd
# df = pd.read_csv('diabetes.csv')
# from sklearn.model_selection import train_test_split
# x = np.array(df.drop(columns='Outcome'))
# y = np.array(df['Outcome'])
# from sklearn.preprocessing import StandardScaler
# scaler = StandardScaler()
# scaler.fit(x)
# x_scaled = scaler.transform(x)
# x_train,x_test,y_train,y_test = train_test_split(x_scaled,y,train_size=0.8)
# from sklearn.linear_model import LogisticRegression
# clf = LogisticRegression()
# clf.fit(x_train,y_train)
# y_pred = clf.predict(x_test)
# print(y_pred)
# import joblib
# joblib.dump(clf, 'diabetes_model.joblib')



#For Breast Cancer
#import numpy as np
#import pandas as pd
#df = pd.read_csv('breast_cancer.csv')
#from sklearn.model_selection import train_test_split
#from sklearn.ensemble import RandomForestClassifier
#from sklearn.metrics import accuracy_score
#df=df.drop("id",axis=1)
#X=df.drop("diagnosis",axis=1)
#y=df["diagnosis"]
#X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42)
#rf=RandomForestClassifier(n_estimators=100)
#rf.fit(X_train,y_train)
#y_pred=rf.predict(X_test)
#import joblib
#joblib.dump(rf, 'breastcancer_model.joblib')
