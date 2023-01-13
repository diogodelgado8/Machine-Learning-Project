'''
Machine Learning 21-22
Project Part 1 - Regression

@authors: Group 134
Diogo Delgado, 92676
Mariana Lima, 92707
'''

import pandas as pd
import numpy as np
import sklearn as skl
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LassoCV
from sklearn.linear_model import Lasso
from sklearn.linear_model import RidgeCV
from sklearn.linear_model import Ridge 
from sklearn.linear_model import BayesianRidge
from sklearn.model_selection import LeaveOneOut
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from eval_scores import scores
import matplotlib.pyplot as plt

Xtrain = np.load('Xtrain_Regression_Part1.npy') 
Ytrain = np.load('Ytrain_Regression_Part1.npy')
Xtest = np.load('Xtest_Regression_Part1.npy')

df_describe = pd.DataFrame(Xtrain)
stats_x = df_describe.describe()

df_describe = pd.DataFrame(Ytrain)
stats_y = df_describe.describe()

#print("Size of Xtrain is", Xtrain.shape)  # (100,20)
#print("Size of Ytrain is", Ytrain.shape)  # (100,1)
#print("Size of Xtest is", Xtest.shape)    # (1000,20)

#standard Xtrain and Xtest
Xtrain_std = StandardScaler().fit_transform(Xtrain)
Xtest_std = StandardScaler().fit_transform(Xtest)


#define cross-validation method to use
cvloo = LeaveOneOut()





'''
    Linear Regression + Leave One Out Cross-Validation
'''
#build multiple linear regression model
model = LinearRegression()

#use LOOCV to evaluate model
errors = cross_val_score(model, Xtrain_std, Ytrain, scoring='neg_mean_squared_error',
                         cv=cvloo, n_jobs=-1)
#view mean squared error
print("Linear Regression Mean Squared Error (MSE) is:",np.mean(np.absolute(errors)))





'''
    Bayesian Ridge Regression + Leave One Out Cross-Validation
'''
#build Bayesian Ridge regression model
model = BayesianRidge()
errors = cross_val_score(model, Xtrain_std, np.ravel(Ytrain), scoring='neg_mean_squared_error',
                         cv=cvloo, n_jobs=-1)
print("Bayesian Ridge Regression Mean Squared Error (MSE) is:",np.mean(np.absolute(errors)))





'''
    Ridge Regression + Leave One Out Cross Validation
'''
#vector of alphas for input
array_Ridge = np.arange(0.000001, 0.05, 0.00001)

#build Ridge CV regression model
model = RidgeCV(alphas=array_Ridge).fit(Xtrain_std, np.ravel(Ytrain))

#get Ridge optimal alpha
Ridge_optimal_alpha = model.alpha_
print("Ridge Optimal Alpha = ", Ridge_optimal_alpha)

model1 = Ridge(alpha=Ridge_optimal_alpha)
errors = cross_val_score(model1, Xtrain_std, Ytrain, scoring='neg_mean_squared_error',
                         cv=cvloo, n_jobs=-1)
print("Ridge CV Regression Mean Squared Error (MSE) is:",np.mean(np.absolute(errors)))





'''
    Lasso Regression + Leave One Out Cross Validation
'''
#build Lasso CV regression model
model = LassoCV(cv=cvloo, random_state=0).fit(Xtrain_std, np.ravel(Ytrain))
#get Lasso optimal alpha
Lasso_optimal_alpha = model.alpha_
print("Lasso Optimal Alpha = ", Lasso_optimal_alpha)

model1 = Lasso(alpha=Lasso_optimal_alpha)
errors = cross_val_score(model1, Xtrain_std, Ytrain, scoring='neg_mean_squared_error',
                         cv=cvloo, n_jobs=-1)
print("Lasso CV Regression Mean Squared Error (MSE) is:",np.mean(np.absolute(errors)))


print("Best model: Lasso with Leave One Out Cross-Validation")



'''
    Best Model - Lasso(alpha = optimal alpha) Overfitting?
'''

Xtrain_split, Xtest_split, Ytrain_split, Ytest_split = train_test_split(Xtrain_std, Ytrain, test_size=0.30, random_state=42)

model1.fit(Xtrain_split, np.ravel(Ytrain_split))
Ytrain_predict = model1.predict(Xtrain_split)
Ytest_predict = model1.predict(Xtest_split)

print("Training Data:") 
scores(np.ravel(Ytrain_split),Ytrain_predict,'r')
print("Test Data:")
scores(np.ravel(Ytest_split),Ytest_predict,'r')





'''
    Final Result - Ytest predicted with the best model (Lasso CV)
'''
Ytest = model.predict(Xtest_std)
np.save('Ytest_Regression_Part1_G134', Ytest, allow_pickle=True, fix_imports=True)







