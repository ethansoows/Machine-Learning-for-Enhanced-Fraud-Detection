# -*- coding: utf-8 -*-
"""
@author: wsoo
"""

#data https://www.kaggle.com/ntnu-testimon/paysim1

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, learning_curve
from sklearn.metrics import average_precision_score
from xgboost.sklearn import XGBClassifier
from xgboost import plot_importance, to_graphviz
from sklearn.preprocessing import OneHotEncoder



df = pd.read_csv('C:\\Users\\WSOO\\Desktop\\Score rating - AML\\PS_20174392719_1491204439457_log.csv')
pd.set_option('display.max_columns', None)
df = df.rename(columns={'oldbalanceOrg':'oldBalanceOrig', 'newbalanceOrig':'newBalanceOrig', \
                        'oldbalanceDest':'oldBalanceDest', 'newbalanceDest':'newBalanceDest'})


#Data exploratory
#1. What are the types of money trander for fraud?
#1.1 All types of transfers 
df['type'].unique()

'''
array(['PAYMENT', 'TRANSFER', 'CASH_OUT', 'DEBIT', 'CASH_IN'],
      dtype=object)
'''

#1.2 Type of transfers that are fraudulent, and what are their counts?
df[df['isFraud'] ==1]['type'].unique()

'''
 array(['TRANSFER', 'CASH_OUT'], dtype=object)
'''

#1.3 Count of these fradulent transactions
df[df['isFraud'] ==1].groupby('type')['type'].count()
'''
type
CASH_OUT    4116
TRANSFER    4097
Name: type, dtype: int64
'''
#we can see that only a small set of data is fradulent - approx 8k our of 6mil



#2. what is the difference between isFraud and isFlaggedFraud? 
'''
isFraud - These are transactions made by the fraudulent agents inside the simulation. In this specific dataset 
the fraudulent behavior of the agents aims to profit by taking control or customers accounts 
and try to empty the funds by transferring to another account and then cashing out of the system.


isFlaggedFraud - The business model aims to control massive transfers 
from one account to another and flags illegal attempts. 
An illegal attempt in this dataset is an attempt to 
transfer more than 200.000 in a single transaction.


Therefore i want to check how many rows there for isFlaggedFraud are and if the definition for isFlaggedFraud is true.
'''

df.loc[df['isFlaggedFraud'] == 1].count()
#16 rows in total

df.loc[( df['amount'] >= 200000.0) & (df['type'] == 'TRANSFER')].count()
#definition of isFlaggedFraud seem to be incorrect as 
#there are 409,110 rows, and not 16 actually flagged.

#also, let me check the max amount where isFlaggedFraud = 0
df.loc[(df['isFlaggedFraud'] ==0)]['amount'].max()
#92445516.64 this also go against the definition. These values might be generated during the simulation process. 

'''
I am going to remove the column since it is not meaningful
'''

df.drop(['isFlaggedFraud'], axis = 1, inplace=True) 


#2.1
#I CAN DO MORE EXPLORATORY DATA ANALYSIS BUT AM GOING TO LEAVE IT FOR NOW. SINCE THIS IS A TRIAL RUN. IMPT - COME BACK AT A LATER DATE. 



##3 Data Cleaning and splitting them to features and lables
X =   df.loc[df['type'].isin(['TRANSFER', 'CASH_OUT'])]

#there are 2770409 data
y = X.isFraud


#3.1 Encoding 'type' to 1 and 0
from sklearn.preprocessing import LabelEncoder
X.iloc[:, 1] = LabelEncoder().fit_transform(X.type)
'''
type - 'transfer' = 1
type - 'CASH_OUT' = 2
'''

#3.2 Name - we need to create a sparse matrix if we are to use name. 
#Therefore, identify if distinct name appear on multiple occasions -is yes, it is significant, if not it discard. 
X['nameOrig'].nunique()
#2768630


#3.2.1 nameOrig - outgoing funds
Dup_nameOrig = X[X.duplicated(['nameOrig'])]['nameOrig'].unique()
#len(Dup_nameOrig) - There are 1776 nameOrig which made more than 1 outgoing tranfer. 
#Are they fradulent? If yes how many outgoing are fraudulent? 

Xfraud = X.loc[y == 1]
Xnonfraud = X.loc[y == 0]

nameOrig_morethan1andFraud = []    
for index, row in Xfraud.iterrows():
    if row['nameOrig'] in Dup_nameOrig:
            #print(row['nameOrig'])
            nameOrig_morethan1andFraud.append(row['nameOrig'])    
print (nameOrig_morethan1andFraud)
#A quick check shows that they only appear once. Not a repeating pattern. 
            
#3.2.2 nameOrig - outgoing funds
Dup_nameDest= X[X.duplicated(['nameDest'])]['nameDest'].unique()
#len(Dup_nameDest) - There are 381,234 nameDest which received more than 1 transfer.
#Are they fradulent? If yes how many are fradulent?
nameDest_morethan1andFraud = []    
for index, row in Xfraud.iterrows():
    if row['nameDest'] in Dup_nameDest:
            #print(row['nameDest'])
            nameDest_morethan1andFraud.append(row['nameDest'])    
len(nameDest_morethan1andFraud) - 5111
len(list(nameDest_morethan1andFraud)) - 5111

'''
Interestingly, 5111 out of 8213 of the fraudulent transactions' recepients received funds more than once(once fraud and once or more non fraud)#

Since nameDest/recepient for fradulent transactions occurs only once, this justifies the removel of this column
as there are no reoccurance and provides no additional information. 

In summary, name as a column used in modelling is a weak feature since this
will create a sparse matrix - leading to curse of dimensionality.
Additionally, in fraud detection, if a name has been flagged, it will be labled as fraud
in any transactions involving that party no matter the type of transaction.
'''


X.drop(['isFraud', 'nameDest', 'nameOrig'], axis = 1, inplace=True) 





#4.Artificial balancing. 
#https://towardsdatascience.com/methods-for-dealing-with-imbalanced-data-5b761be45a18

'''
Problems with imbalanced data.

Since PaySim data was synthetically generated. It will be useful to remove data 
from dominant class to make it more balanced. In fact, we can make it to the real ratio e.g. in Stenson project,
say if the number of true positive is flagged is 40%, we simulate that.  For this example, i used 66% no fraud

Further, it is wise to focus on 'Transfers' and 'Cashouts' since isFraud only happens at here
'''


#4.1 undersample training data
#undersampling to remove data from majority class
from sklearn.utils import resample


# setting up testing and training sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=27)




# concatenate our training data back together
Training_df = pd.concat([X_train, y_train], axis=1)

# separate minority and majority classes on training data
not_fraud = Training_df[Training_df.isFraud==0]
fraud = Training_df[Training_df.isFraud==1]

#as mentioned above, i want to undersample the data
not_fraud_downsampled = resample(not_fraud,
                                replace = False, # sample without replacement
                                n_samples = round(len(fraud)*2), # this is where i can set the ratio to be realistic
                                random_state = 27) 

X_downsampled = pd.concat([not_fraud_downsampled, fraud])
X_downsampled.isFraud.value_counts()
#how there are 12226 and 6113 for the 2 categories above respectively 
#0    12226
#1     6113


#assigning them back to their same classes
y_train = X_downsampled.isFraud
X_train = X_downsampled.drop('isFraud', axis=1)

#4.2 undersampling testing data

# concatenate our testing data back together
Testing_df = pd.concat([X_test, y_test], axis=1)

# separate minority and majority classes on testing data
not_fraud_testing = Testing_df[Testing_df.isFraud==0]
fraud_testing = Testing_df[Testing_df.isFraud==1]

#as mentioned above, i want to undersample the data
not_fraud_downsampled_testing = resample(not_fraud_testing,
                                replace = False, # sample without replacement
                                n_samples = round(len(fraud_testing)*2), # this is where i can set the ratio to be realistic
                                random_state = 27) 

X_downsampled_testing = pd.concat([not_fraud_downsampled_testing, fraud_testing])
X_downsampled_testing.isFraud.value_counts()
#now there are 4200 and 2100 for the 2 categories above respectively 


#assigning them back to their same classes
y_test = X_downsampled_testing.isFraud
X_test = X_downsampled_testing.drop('isFraud', axis=1)



#5 Which supervised classfiers are most suitable?
#https://web.archive.org/web/20140311005243/http://machinelearning.wustl.edu/mlpapers/paper_files/icml2005_Niculescu-MizilC05.pdf
#Summary of puclication above - boosted trees, random forest and SVM are most suitable. For more details on why - please look into article.


#5.1.1 boosted trees
#https://machinelearningmastery.com/develop-first-xgboost-model-python-scikit-learn/
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score

# fit model no training data
modelXGB = XGBClassifier()
modelXGB.fit(X_train, y_train)

y_pred = modelXGB.predict(X_test)
np.unique(y_pred)
import collections
collections.Counter(y_pred)
'''
counter({0: 679524, 1: 13079})
'''

	
# evaluate predictions
from sklearn.metrics import confusion_matrix


accuracy = accuracy_score(y_test, y_pred)
cmXGB = confusion_matrix(y_test, y_pred)
print('Confusion matrix:','\n',  cmXGB)
print("Accuracy: %.2f%%" % (accuracy * 100.0))
tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
print("True Negatives: ",tn)
print("False Positives: ",fp)
print("False Negatives: ",fn)
print("True Positives: ",tp)

#we do not have to worry about prevision an recall since we train and test it on somewhat balanced dataset. 

#what happens if i try to run it on the whole dataset? 





'''
old - training data not undersamples to make it equal: testing it on not undersampled data
Confusion metrix: 
 [[679494  11009]
 [    30   2070]]
 
 new - training data undersampled: testing it on undersampled data
 [[4133   67]
 [  30 2070]]
 
 
 
Accuracy: 98.41%
'''


#5.1.2 Using Predit_Proba
#https://datascience.stackexchange.com/questions/8032/how-to-predict-probabilities-in-xgboost
y_pred_proba = modelXGB.predict_proba(X_test)






#5.1.3 comparing 2 examples above

combined = np.column_stack((y_pred,y_pred_proba))
combined = pd.DataFrame(combined,  columns = ['y_pred_absolute', 'y_pred_proba_0', 'y_pred_proba_1'])

y_test_df = pd.DataFrame(y_test, columns = ['isFraud'])
y_test_df.reset_index(drop=True, inplace=True)
#X_test_reset_index = X_test.reset_index(drop=True, inplace=True)


X_test.reset_index(drop=True, inplace=True)
y_test_df.reset_index(drop=True, inplace=True)

full_test_results = pd.concat([X_test, y_test_df, combined ], axis = 1)
#https://python-forum.io/Thread-Join-Predicted-values-with-test-dataset order is kept (good)


'''
QC

full_results.loc[(full_results['isFraud'] == 1)]
#2100

full_results.loc[(full_results['y_pred_absolute'] == 0.0 )].count()
#correct - 679494 + 30 = 679524  
'''

#5.1.4 Plotting a histogram into 10 groups will be useful to see how the 1s are spread. 

false_negative = full_test_results.loc[(full_test_results['isFraud'] == 1) & (full_test_results['y_pred_absolute'] == 0.0 )]

false_positive= full_test_results.loc[(full_test_results['isFraud'] == 0) & (full_test_results['y_pred_absolute'] == 1.0 )]


false_positive.hist(column = 'y_pred_proba_1', bins = 5)
false_negative.hist(column = 'y_pred_proba_0', bins = 5)

full_test_results.hist(column = 'y_pred_proba_1', bins = 20)



'''
From the plot above, we can see that majority of them falls under 0.0 - 0.1 catrgory. 
This is commensurate with what we see in the data exploratory. 
'''

'''
In a real world situation,we know that there will be many false negative. this is because most of the transactions are negative, 
and wrong prediction(small percentage) of a large count will still result in a large number

On the other note, we want to limit the number of false negative - because these are laundered money!!!!
Through this exercise, even though the number is small, 30, ideally we want to remove it to 0. 

'''



#5.1.4 Platt Scaling #https://machinelearningmastery.com/calibrated-classification-model-in-scikit-learn/

#There are two concerns in calibrating probabilities; they are diagnosing the calibration of predicted 
#probabilities and the calibration process itself.
#Platt Scaling is simpler and is suitable for reliability diagrams with the S-shape. Isotonic Regression is more complex, requires a lot more data (otherwise it may overfit), but can support reliability diagrams with different shapes (is nonparametric).


#5.1.4.1 Draw Reliability Diagram to diagnose calibration
import sklearn.calibration 
import matplotlib
y_pred_proba_1 = [el[1] for el in  y_pred_proba]
fop, mpv = sklearn.calibration.calibration_curve(y_test, y_pred_proba_1, n_bins=10)
# plot perfectly calibrated
matplotlib.pyplot.plot([0, 1], [0, 1], linestyle='--')
# plot model reliability
matplotlib.pyplot.plot(mpv, fop, marker='.')
matplotlib.pyplot.show()



#5.1.4.2 Calibrate Classifier
from sklearn.calibration import CalibratedClassifierCV
calibrated  = CalibratedClassifierCV(modelXGB, method='sigmoid', cv=3) # 
#isotonic is not useful because it is the original disribution is sigmoid shaped
#used cv = 3,5,8,10. 3 seems to provide the closest shape to reliability line for probablility equals 1
calibrated.fit(X_train, y_train)
#predict probabilities
y_calibrated = calibrated.predict_proba(X_test)
y_calibrated_1 = [el[1] for el in  y_calibrated]
# reliability diagram
fop, mpv = sklearn.calibration.calibration_curve(y_test, y_calibrated_1, n_bins=10)
# plot perfectly calibrated
matplotlib.pyplot.plot([0, 1], [0, 1], linestyle='--')
# plot calibrated reliability
matplotlib.pyplot.plot(mpv, fop, marker='.')
matplotlib.pyplot.show()



#getting the probability of 1 only - for comparison
y_calibrated_1 =  pd.DataFrame(y_calibrated_1, columns = ['y_calibrated_proba_1'])

#The CalibratedClassifierCV class supports two types of probability calibration; specifically, the parametric ‘sigmoid‘ method (Platt’s method) and the nonparametric ‘isotonic‘ method which can be specified via the ‘method‘ argument.


#join all the results 
full_test_results_calibrated = pd.concat([full_test_results, y_calibrated_1], axis = 1 )

# i need to give y_calibrated_proba_1 an absolute number before i can test accuracy


false_negative_calibrated = full_test_results_calibrated.loc[(full_test_results_calibrated['isFraud'] == 1) & (full_test_results_calibrated['y_calibrated_proba_1'] <= 0.49999 )]

false_positive_calibrated = full_test_results_calibrated.loc[(full_test_results_calibrated['isFraud'] == 0) & (full_test_results_calibrated['y_calibrated_proba_1'] >= 0.51111111 )]


false_positive_calibrated.hist(column = 'y_calibrated_proba_1', bins = 5)
false_negative_calibrated.hist(column = 'y_calibrated_proba_1', bins = 5)


'''
calibration makes it worse because of the distortion around 0.4 and 0.5 probability



calibration will not help those sample points where the class classification is wrong 
since calibration will not change move it to the other side

'''


#5.2 Random Forest




#5.3.1 SVM
#https://stats.stackexchange.com/questions/154224/when-using-svms-why-do-i-need-to-scale-the-features
# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_trainSVM = sc.fit_transform(X_train)
X_testSVM = sc.transform(X_test)

#converting it back to dataframe
X_trainSVM  = pd.DataFrame(X_trainSVM, index=X_train.index, columns=X_train.columns)
X_testSVM  = pd.DataFrame(X_testSVM, index=X_test.index, columns=X_test.columns)

from sklearn.svm import SVC
SVM = SVC(kernel = 'linear', random_state = 0)
SVM.fit(X_trainSVM, y_train)

# Predicting the Test set results
y_predSVM = SVM.predict(X_testSVM)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix

cmSVM = confusion_matrix(y_test, y_predSVM)
accuracySVM = accuracy_score(y_test, y_predSVM)
print('Confusion matrix: \n',cmSVM)
print("Accuracy: %.2f%%" % (accuracySVM * 100.0))


#5.3.2 SVM using probability



#5.3.3 Comparing he 2 models




#6 Predicting probabilities of the classifiers
#https://stackoverflow.com/questions/30814231/using-the-predict-proba-function-of-randomforestclassifier-in-the-safe-and-rig


#7 Calibrating probabilities 
#https://machinelearningmastery.com/calibrated-classification-model-in-scikit-learn/


'''
Question to ponder on. Do transfer nameDest and cashout nameOrig matches? They should match. However this does not give predictability 
since isFraud column is populated after Fraud took place. 
'''








