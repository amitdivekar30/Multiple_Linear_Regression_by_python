# Multiple Linear Regression
# Predict Price of the computer 
#importing libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#Importing Dataset
computer=pd.read_csv('Computer_Data.csv')
computer.columns
computer=computer.iloc[:, 1: ]
computer.columns

# to get top 10 rows
computer.head(10) 

# Correlation matrix 
computer.corr()
 
# Scatter plot between the variables along with histograms
import seaborn as sns
sns.pairplot(computer)

# Encoding categorical data
# Encoding the Independent Variable
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder = LabelEncoder()
for i in [5,6,7]:
    computer.iloc[:, i] = labelencoder.fit_transform(computer.iloc[:, i])
#    onehotencoder = OneHotEncoder(i)
#    computer = onehotencoder.fit_transform(computer).toarray()


# preparing model considering all the variables 
import statsmodels.formula.api as smf # for regression model
         
# Preparing model                  
regressor = smf.ols('price~ speed+hd+ram+screen+cd+multi+premium+ads+trend',data=computer).fit() # regression model
# Getting coefficients of variables        
regressor.params
# Summary
regressor.summary() #0.776

# Checking whether data has any influential values 
# influence index plots
import statsmodels.api as sm
sm.graphics.influence_plot(regressor)

# index 1400 AND 1740 is showing high influence so we can exclude that entire row
computer_new=computer.drop(computer.index[[1400,1740]],axis=0)

# Preparing model                  
regressor1 = smf.ols('price~ speed+hd+ram+screen+cd+multi+premium+ads+trend',data=computer_new).fit() # regression model
# Getting coefficients of variables        
regressor1.params
# Summary
regressor1.summary() #0.776

# Confidence values 99%
print(regressor1.conf_int(0.01)) # 99% confidence level


# Predicted values of MPG 
price_pred = regressor1.predict(computer_new.iloc[: ,1: ])
price_pred

# calculating VIF's values of independent variables
rsq_speed = smf.ols('speed~hd+ram+screen+cd+multi+premium+ads+trend',data=computer_new).fit().rsquared  
vif_speed = 1/(1-rsq_speed) # 1.26

rsq_hd = smf.ols('hd~speed+ram+screen+cd+multi+premium+ads+trend',data=computer_new).fit().rsquared  
vif_hd = 1/(1-rsq_hd) # 4.21

rsq_ram = smf.ols('ram~hd+speed+screen+cd+multi+premium+ads+trend',data=computer_new).fit().rsquared  
vif_ram = 1/(1-rsq_ram) # 2.97

rsq_screen = smf.ols('screen~hd+ram+speed+cd+multi+premium+ads+trend',data=computer_new).fit().rsquared  
vif_screen = 1/(1-rsq_screen) # 1.081

rsq_multi = smf.ols('multi~hd+ram+screen+cd+speed+premium+ads+trend',data=computer_new).fit().rsquared  
vif_multi = 1/(1-rsq_multi) # 1.29

rsq_premium = smf.ols('premium~hd+ram+screen+cd+multi+ speed +ads+trend',data=computer_new).fit().rsquared  
vif_premium = 1/(1-rsq_premium) # 1.11

rsq_ads = smf.ols('ads~hd+ram+screen+cd+multi+premium+ speed +trend',data=computer_new).fit().rsquared  
vif_ads = 1/(1-rsq_ads) # 1.21

rsq_trend = smf.ols('trend~hd+ram+screen+cd+multi+premium+ads+ speed',data=computer_new).fit().rsquared  
vif_trend = 1/(1-rsq_trend) # 2.022

rsq_cd = smf.ols('cd~hd+ram+screen+trend+multi+premium+ads+ speed',data=computer_new).fit().rsquared  
vif_cd = 1/(1-rsq_cd) #1.896

           # Storing vif values in a data frame
d1 = {'Variables':['speed','hd' ,'ram' ,'screen' ,'cd' ,'multi' ,'premium' ,'ads' ,'trend'],'VIF':[vif_speed, vif_hd, vif_ram, vif_screen, vif_cd, vif_multi, vif_premium, vif_ads, vif_trend]}
Vif_frame = pd.DataFrame(d1)  
Vif_frame

# Added varible plot 
sm.graphics.plot_partregress_grid(regressor1)

# final model
final_model= smf.ols('price~ speed+hd+ram+screen+cd+multi+premium+ads+trend',data = computer_new).fit()
final_model.params
final_model.summary() # 0.747

price_pred = final_ml.predict(cars_new)

import statsmodels.api as sm
# added variable plot for the final model
sm.graphics.plot_partregress_grid(final_model)


######  Linearity #########
# Observed values VS Fitted values
plt.scatter(computer_new.price,price_pred,c="r");plt.xlabel("observed_values");plt.ylabel("fitted_values")

# Residuals VS Fitted Values 
plt.scatter(price_pred,final_model.resid_pearson,c="r"),plt.axhline(y=0,color='blue');plt.xlabel("fitted_values");plt.ylabel("residuals")


########    Normality plot for residuals ######
# histogram
plt.hist(final_model.resid_pearson) # Checking the standardized residuals are normally distributed

# QQ plot for residuals 
import pylab          
import scipy.stats as st

# Checking Residuals are normally distributed
st.probplot(final_model.resid_pearson, dist="norm", plot=pylab)


############ Homoscedasticity #######

# Residuals VS Fitted Values 
plt.scatter(price_pred,final_model.resid_pearson,c="r"),plt.axhline(y=0,color='blue');plt.xlabel("fitted_values");plt.ylabel("residuals")


#spilitting into X and Y
X=computer_new.iloc[:, 1:].values #all the columns except last one
Y=computer_new.iloc[:, 0].values # all the columns in index 0

#Splitting the Dataset into Training Set and Test Set
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test=train_test_split(X, Y, test_size=0.2, random_state=0)

#Fitting Multiple Linear Regression to the training set
from sklearn.linear_model import LinearRegression
regressor2= LinearRegression()
regressor2.fit(X_train, Y_train)

#predicting the Train Set Results
price_pred_train=regressor2.predict(X_train)
#predicting the Test Set Results
price_pred_test=regressor2.predict(X_test)


# train residual values 
train_resid  = price_pred_train - Y_train

# RMSE value for train data 
train_rmse = np.sqrt(np.mean(train_resid*train_resid))

# prediction on test data set 
test_pred = model_train.predict(cars_test)

# test residual values 
test_resid  =price_pred_test- Y_test
# RMSE value for test data 
test_rmse = np.sqrt(np.mean(test_resid*test_resid))
