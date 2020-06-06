import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Read the datset
dataset = pd.read_csv('dataset.csv')

# View the dataset
dataset.info()

# You can view the dataset by in 'sample' dataframe
sample = dataset.head(10)

# X => abtest,vahicletype,year_of_registration,gearbox,powerPS,model,kilometer,month_of_registration,fueltype,brand,not_repair_damege, date_create, postal_code,lastseen
X = dataset.iloc[:,3:16].values

# y => price
y = dataset.iloc[:,2:3].values

# load the column no which have to convart from string to  by label encoder
ind = [0,1,3,5,8,9,10]

# Excecute label encoder
from sklearn.preprocessing import LabelEncoder
for i in ind:
    labelencoder = LabelEncoder()
    X[:,i]=X[:,i].astype(str)
    X[:,i]=labelencoder.fit_transform(X[:,i])

# Split the dataset 80:20 ratio
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state =40)


# Consider mean_absolute_error and explain_varience_score as measurement unit of our project
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import explained_variance_score

# array, that store mean_absolute_error of all regression model
mae = []

# array, that store explain_varience_score of all regression model
evs = []


###############################################################################
## Linear Regression

# Training
from sklearn.linear_model import LinearRegression ## import library
regressor1 = LinearRegression() ## create object
regressor1.fit(X_train, y_train) # train

y_pred1 = regressor1.predict(X_test) ## test

mae.append(mean_absolute_error(y_test, y_pred1)) ## calculate mean_absolute_error and store in 'mae' array
evs.append(explained_variance_score(y_test, y_pred1)) ## calculate explain_varience_score and store in 'evs' array

###############################################################################
## Decision Tree

from sklearn.tree import DecisionTreeRegressor
regressor2 = DecisionTreeRegressor(random_state = 0)
regressor2.fit(X_train, y_train)

y_pred2 = regressor2.predict(X_test)

mae.append(mean_absolute_error(y_test, y_pred2))
evs.append(explained_variance_score(y_test, y_pred2))

###############################################################################
## Random Forest

from sklearn.ensemble import RandomForestRegressor
regressor3 = RandomForestRegressor(n_estimators = 10, random_state = 0)
regressor3.fit(X_train, y_train.ravel())  # ravel() used for convert to 1d array

y_pred3 = regressor3.predict(X_test) ##test

mae.append(mean_absolute_error(y_test, y_pred3))
evs.append(explained_variance_score(y_test, y_pred3))

###############################################################################
## Support Vector Machine (SVM) / Support Vector Regressor (SVR)

# Scalling the data (i.e. X and y) in uniform formate by standard scaler
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
sc_y = StandardScaler()
X = sc_X.fit_transform(X)
y = sc_y.fit_transform(np.reshape(y, (len(y), 1)))

# Then the scaling data split in train and test set in 80:20 as previous
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state =40)

from sklearn.svm import SVR
regressor4 = SVR(kernel = 'rbf')
regressor4.fit(X_train, y_train.ravel())

y_pred4 = regressor4.predict(X_test)

# Inverse scaling for accurate measurement
y_testt=sc_y.inverse_transform(y_test)
y_predd=sc_y.inverse_transform(y_pred4)

mae.append(mean_absolute_error(y_testt, y_predd))
evs.append(explained_variance_score(y_testt, y_predd))

###############################################################################

print("Run Complete!")
## you can view the score of the measurment unit (mae and evs) in variable explorer
