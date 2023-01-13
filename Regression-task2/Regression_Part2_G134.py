'''
Machine Learning 21-22
Project Part 2 - Regression

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
from sklearn.linear_model import OrthogonalMatchingPursuitCV
from sklearn.linear_model import OrthogonalMatchingPursuit
from sklearn.model_selection import LeaveOneOut
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from eval_scores import scores

# Load data
Xtrain = np.load('Xtrain_Regression_Part2.npy') 
Ytrain = np.load('Ytrain_Regression_Part2.npy')
Xtest = np.load('Xtest_Regression_Part2.npy')


# Maxim number of outliers is less than 10% of data sample 
MaxOutliers = int(0.0999*Xtrain.shape[0]) # 9

# See statistics of data
df_describe = pd.DataFrame(Xtrain)
Xstats = df_describe.describe()

df_describe = pd.DataFrame(Ytrain)
Ystats = df_describe.describe()

# See data shape
#print("Size of Xtrain is", Xtrain.shape)  #(100,20)
#print("Size of Ytrain is", Ytrain.shape)  #(100,1)
#print("Size of Xtest is", Xtest.shape)    #(1000,20)

#define cross-validation method to use
cvloo = LeaveOneOut()


################################################################################################    
#   Function Remove Outliers
#       Inputs: Training data (Xtrain,Ytrain) & Maximum number of outliers  
#       Outputs: Training Data without outliers (XtrainOutliers,YtrainOutliers) & number of outliers detected 
################################################################################################

def RemoveOutliers (Xtrain, Ytrain, MaxOutliers):

    XtrainOutliers = Xtrain
    YtrainOutliers = Ytrain
    
    # Initialize local variables
    MaxIter = int(MaxOutliers*2)
    GlobalError = np.zeros(MaxIter)
    MaxError = np.zeros(MaxIter)
    MaxErrorIndex = np.zeros(MaxIter, dtype=int)
    ErrorsDif = np.zeros(MaxIter-1)
 
    # Model assumed for outlier detection
    model = LinearRegression()
    
    # For loop 
    for i in range(MaxIter):
        #use LOOCV to evaluate model
        errors = cross_val_score(model, XtrainOutliers, YtrainOutliers, scoring='neg_mean_squared_error', cv=cvloo, n_jobs=-1)

        TotalErrors = np.absolute(errors)
        #print("Outliers removed: ", i)
        MaxError[i] = np.amax(TotalErrors)
        #print("MSE máximo local:", MaxError[i])
    
        MaxErrorIndex[i] = np.argmax(TotalErrors)
        #print("Índice Outlier", MaxErrorIndex[i])   
        
        GlobalError[i] = np.mean(TotalErrors)
        #print("MSE Global:", GlobalError[i])
        #print()
        
        # Compute the difference between global MSEs 
        if i!=0:
            ErrorsDif[i-1] = GlobalError[i-1] - GlobalError[i]
        
        # Remove row (in X and Y) with the biggest MSE
        XtrainOutliers = np.delete(XtrainOutliers, MaxErrorIndex[i], 0)
        YtrainOutliers = np.delete(YtrainOutliers, MaxErrorIndex[i], 0)
    
    # Calculate mean and max of ErrorsDif in iterations where it's certain that no outliers exist
    # to discover the pattern of differences between MSEs
    MeanErrorsDif = np.mean(np.absolute(ErrorsDif[MaxOutliers+1:]))
    
    #print(MeanErrorsDif)
    
    
    # Check the number of outliers
    NumOutliers = 0
    for i in range(MaxOutliers):
        if ErrorsDif[i] > 1.5*MeanErrorsDif: # Criteria to select outliers (+50% of MeanErrorsDif)
            NumOutliers = NumOutliers+1
        else:
            break
    
    # Eliminate outliers
    XtrainOutliers = Xtrain
    YtrainOutliers = Ytrain
        
    for i in range(NumOutliers):
        XtrainOutliers = np.delete(XtrainOutliers, MaxErrorIndex[i], 0)
        YtrainOutliers = np.delete(YtrainOutliers, MaxErrorIndex[i], 0)
    
    
    return XtrainOutliers, YtrainOutliers, NumOutliers


################################################################################################
#   Train "correct data" (withouth outliers)
################################################################################################

XtrainOutliers, YtrainOutliers, NumOutliers = RemoveOutliers(Xtrain, Ytrain, MaxOutliers)
scaler = StandardScaler().fit(XtrainOutliers)
XtrainOutliers = scaler.transform(XtrainOutliers) 
Xtest = scaler.transform(Xtest)

print("Detected", NumOutliers, "outliers")
print()
print("Check MSEs for 5 Linear Models...")
print()

################################################################################################
#   Linear Regression + Leave One Out Cross-Validation
################################################################################################

#build multiple linear regression model
model = LinearRegression()

#use LOOCV to evaluate model
errors = cross_val_score(model, XtrainOutliers, YtrainOutliers, scoring='neg_mean_squared_error',
                         cv=cvloo, n_jobs=-1)
#view mean squared error
print("Linear Regression:") 
print("   Mean Squared Error (MSE) is",np.mean(np.absolute(errors)))
print()


################################################################################################
#   Bayesian Ridge Regression + Leave One Out Cross-Validation
################################################################################################

#build Bayesian Ridge regression model
model = BayesianRidge()
errors = cross_val_score(model, XtrainOutliers, np.ravel(YtrainOutliers), scoring='neg_mean_squared_error',
                         cv=cvloo, n_jobs=-1)
print("Bayesian Regression:") 
print("   Mean Squared Error (MSE) is",np.mean(np.absolute(errors)))
print()

################################################################################################
#   Ridge Regression + Leave One Out Cross Validation
################################################################################################

#vector of alphas for input
array_Ridge = np.arange(0.0000001, 0.005, 0.0000001)

#build Ridge CV regression model
model = RidgeCV(alphas=array_Ridge).fit(XtrainOutliers, np.ravel(YtrainOutliers))

#get Ridge optimal alpha
Ridge_optimal_alpha = model.alpha_

model1 = Ridge(alpha=Ridge_optimal_alpha)
errors = cross_val_score(model1, XtrainOutliers, YtrainOutliers, scoring='neg_mean_squared_error',
                         cv=cvloo, n_jobs=-1)
print("Ridge Regression:") 
print("  Ridge Optimal Alpha is", Ridge_optimal_alpha)
print("   Mean Squared Error (MSE) is",np.mean(np.absolute(errors)))
print("   Note: This model is choosing the minimum alpha possible. This is equivalent to Linear Regression.")
print()


################################################################################################
#   Lasso Regression + Leave One Out Cross Validation
################################################################################################

#build Lasso CV regression model
model = LassoCV(cv=cvloo, random_state=0).fit(XtrainOutliers, np.ravel(YtrainOutliers))
#get Lasso optimal alpha
Lasso_optimal_alpha = model.alpha_
alphas = model.alphas_

model1 = Lasso(alpha=Lasso_optimal_alpha)
errors = cross_val_score(model1, XtrainOutliers, YtrainOutliers, scoring='neg_mean_squared_error',
                         cv=cvloo, n_jobs=-1)
print("Lasso Regression:") 
print("  Lasso Optimal Alpha is", Lasso_optimal_alpha)
print("   Mean Squared Error (MSE) is",np.mean(np.absolute(errors)))
print()

################################################################################################
#   Orthogonal Matching Persuit + Leave One Out Cross Validation
################################################################################################

#build OMP CV regression model
model = OrthogonalMatchingPursuitCV(cv=cvloo, normalize=False, max_iter=Xtrain.shape[1]).fit(XtrainOutliers, np.ravel(YtrainOutliers))
#get OMP optimal hyperparameter
Omp_optimal_hyp = model.n_nonzero_coefs_

model1 = OrthogonalMatchingPursuit(n_nonzero_coefs=Omp_optimal_hyp, normalize=False)
errors = cross_val_score(model1, XtrainOutliers, YtrainOutliers, scoring='neg_mean_squared_error',
                         cv=cvloo, n_jobs=-1)
print("Orthogonal Matching Persuit Regression:")
print("  Orthogonal Optimal Hyperparameter is", Omp_optimal_hyp)
print("   Mean Squared Error (MSE) is",np.mean(np.absolute(errors)))
print()


################################################################################################
#   Best Model - Orthogonal Matching Persuit (Overfitting?)
################################################################################################
print("The best model is Orthogonal Matching Persuit.")
print()

XtrainSplit, XtestSplit, YtrainSplit, YtestSplit = train_test_split(XtrainOutliers, YtrainOutliers, test_size=0.30, random_state=42)


model = OrthogonalMatchingPursuit(n_nonzero_coefs=Omp_optimal_hyp, normalize=False)

model.fit(XtrainSplit, np.ravel(YtrainSplit))
YtrainPredict = model.predict(XtrainSplit)
YtestPredict = model.predict(XtestSplit)

print("Ckeck is there is overfitting...")
print("Training Data:") 
scores(np.ravel(YtrainSplit),YtrainPredict,'r')
print("Test Data:")
scores(np.ravel(YtestSplit),YtestPredict,'r')



################################################################################################
#  Predict Ytest and save it as a .npy array
################################################################################################

Ytest = model.predict(Xtest)
np.save('Ytest_Regression_Part2_G134', Ytest, allow_pickle=True, fix_imports=True)
