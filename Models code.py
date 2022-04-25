# -*- coding: utf-8 -*-
"""
Created on Sun Apr 10 13:22:26 2022

@author: zoheirahmed-m
"""

import pandas as pd
import numpy              as np

from AdvancedAnalytics.ReplaceImputeEncode import DT, ReplaceImputeEncode

#Linear regression
from AdvancedAnalytics.Regression                 import linreg
from sklearn.linear_model                         import LinearRegression
from sklearn                                      import linear_model
from scipy.special                                import expit

from math                                         import log, isfinite, sqrt, pi
from sklearn.metrics                              import r2_score

# classes for logistic regression
from sklearn.linear_model                         import LogisticRegression

#for Decision Tree
from AdvancedAnalytics.Tree                       import tree_classifier
from sklearn.tree                                 import DecisionTreeClassifier

#for Random Forest
from AdvancedAnalytics.Forest                     import forest_classifier  
from sklearn.ensemble                             import RandomForestClassifier , RandomForestRegressor

#other Libraries  
from AdvancedAnalytics.ReplaceImputeEncode        import DT, ReplaceImputeEncode  #for data processing
from sklearn.model_selection                      import cross_validate , train_test_split  #for Model validation and partioning
from AdvancedAnalytics.Regression                 import linreg, logreg, stepwise #for logistic regression & Stepwise
import statsmodels.api    as sm
import statsmodels.tools.eval_measures as em  


from scipy.stats                                  import norm
import matplotlib.pyplot  as plt
from sklearn.tree                                 import plot_tree
from matplotlib                                   import pyplot as plt
import pickle


import warnings
warnings.filterwarnings("ignore", category=FutureWarning)


df=pd.read_csv('Capstoneprosperselected2.csv')


attribute_map = {
	'ListingNumber': 	                    [DT.ID, (3.5, 1255725.5)],
##	'CreationQtr': 	                        [DT.Nominal,  (1, 2, 3, 4)],
##	'CreationMth': 	                        [DT.Interval, (1, 12)],
##	'CreationYr': 	                        [DT.Interval, (2005, 2014)],
	'CreditGrade': 	                        [DT.Nominal, ('A','AA','B','C','D','E','HR','NC')],
	'Term': 	        	                [DT.Nominal, (12, 36, 60) ],
##  'ClosedYr': 	                        [DT.Interval, (2005, 2014)],
##	'ClosedMth': 	                        [DT.Interval, (1, 12) ],
##  'BorrowerAPR': 	                        [DT.Interval, ( 0.00653, 0.52) ],
##	'BorrowerRate': 	                    [DT.Interval, ( 0.0,  0.5) ],
##	'LenderYield': 	                        [DT.Interval, (-0.01, 0.5) ],
##	'EstimatedEffectiveYield': 	            [DT.Interval, (-0.1827, 0.3199) ],
##	'EstimatedLoss': 	                    [DT.Interval, (-0.4951, 0.866) ],
##	'EstimatedReturn': 	                    [DT.Interval, (-0.1827, 0.2837) ],
# 	'ProsperRatingNumeric':                 [DT.Nominal, (1, 2, 3, 4, 5, 6, 7) ],
# 	'ProsperRatingAlpha': 	                [DT.Nominal, ('A','AA','B','C','D','E','HR','NC')],
# 	'ProsperScore': 	                    [DT.Interval, (1, 11) ],
    'ListingCategory (numeric)':            [DT.Interval, (0.0, 20) ],
	'BorrowerState': 	                    [DT.String,  ],
	'Occupation': 	                        [DT.String,  ],
    'EmploymentStatusDuration':             [DT.Interval, (0.0, 755) ],
#   'GroupKey': 	                        [DT.ID,  ],
##	'CreditPullYr': 	                    [DT.Interval, (2005, 2014) ],
##	'CreditPullQtr': 	                    [DT.Nominal, (1, 2, 3, 4) ],
##	'CreditPullMth': 	                    [DT.Interval, (1, 12) ],
    'CreditScoreRangeLower': 	            [DT.Interval, (0.0, 880) ],
	'CreditScoreRangeUpper': 	            [DT.Interval, (19,  899) ],
##  'FirstCreditLineYr': 	                [DT.Interval, (1947, 2012) ],
##	'FirstCreditLineQtr': 	                [DT.Nominal, (1, 2, 3, 4) ],
##	'FirstCreditLineMth':    	            [DT.Interval, (1, 12) ],
###	'CurrentCreditLines': 	                [DT.Interval, (0.0, 59) ],
### 'OpenCreditLines': 	                    [DT.Interval, (0.0, 54) ],
	'TotalCreditLinespast7years': 	        [DT.Interval, (1.5, 136) ],
	'OpenRevolvingAccounts': 	            [DT.Interval, (0.0, 51.5) ],
	'OpenRevolvingMonthlyPayment': 	        [DT.Interval, (0.0, 14985) ],
##   'InquiriesLast6Months': 	            [DT.Interval, (0.0, 105.5) ],
	'TotalInquiries': 	                    [DT.Interval, (0.0, 379.5) ],
###	'CurrentDelinquencies': 	            [DT.Interval, (0.0, 83.5) ],
	'AmountDelinquent': 	                [DT.Interval, (0.0, 463881.5) ],
##	'DelinquenciesLast7Years': 	            [DT.Interval, (0.0, 99) ],
##	'PublicRecordsLast10Years':             [DT.Interval, (0.0, 38) ],
##	'PublicRecordsLast12Months':            [DT.Nominal, (0, 1, 2, 3, 4, 7, 20) ],
    'RevolvingCreditBalance': 	            [DT.Interval, (0.0, 1435667.5) ],
##  'BankcardUtilization': 	                [DT.Interval, (0.0, 6.45) ],
	'AvailableBankcardCredit': 	            [DT.Interval, (0.0, 646285.5) ],
###	'TotalTrades': 	                        [DT.Interval, (0.0, 126.5) ],
###	'TradesNeverDelinquent (percentage)':   [DT.Interval, (0, 1.0) ],
##	'TradesOpenedLast6Months':          	[DT.Interval, (0, 20) ],
##	'DebtToIncomeRatio':    	            [DT.Interval, (0, 11) ],
    'StatedMonthlyIncome': 	                [DT.Interval, (0.0, 1750003) ],
##	'LoanKey': 	                            [DT.String,  ],
#	'LoanMonthsSinceOrigination': 	        [DT.Interval, (0.0, 100.5) ],
##	'LoanNumber': 	                        [DT.ID, (0.5, 136486.5) ],
	'LoanOriginalAmount': 	                [DT.Interval, (1000, 35000) ],
    'LoanDateYr': 	                        [DT.Interval, (2005, 2014) ],
    'MonthlyLoanPayment': 	                [DT.Interval, (0.0, 2252.01) ],
##	'LoanDateQtr':                      	[DT.Nominal, (1, 2, 3, 4) ],
##	'LoanDateMth': 	                        [DT.Interval, (1, 12) ],
#	'LoanOriginationQuarter':           	[DT.String,  ],
##	'MemberKey': 	                        [DT.ID,  ],
# 	'LP_CustomerPayments': 	                [DT.Interval, (-2.8499, 40702.89)],
# 	'LP_CustomerPrincipalPayments': 	    [DT.Interval, (0.0, 35000.5) ],
# 	'PercentFunded': 	                    [DT.Interval, (0.2, 1.5125) ],
# 	'Recommendations': 	                    [DT.Interval, (0.0, 39.5) ],
    'clean_BinaryDefault':                  [DT.Binary, (0, 1)],
    'clean_IncomeRange':                    [DT.Interval, (0, 21000036) ],
##  'clean_IncomeVerifiable':               [DT.Binary, (0, 1) ],
##   'clean_IsBorrowerHomeowner':            [DT.Binary, (0, 1) ],
#   'clean_CurrentlyInGroup':               [DT.Binary, (0, 1) ],
##  'clean_EmploymentStatus': 	            [DT.String, ('Employed/Full-time','N/A', 'Not employed','Other', 'Part-time', 'Retired') ]
    }


target = 'clean_BinaryDefault'
encoding = 'one-hot'


print("Read", df.shape[0], "observations with", 
      df.shape[1], "attributes:\n")


RIE_Non_Reg = True
Run_Tree    = False 
DecisionTree_run = False
Forest_CV   = True
Forest_Eval = True

if RIE_Non_Reg:
    scale    = None  # Interval scaling:  Use 'std', 'robust' or None
    scaling  = 'No'  # Text description for interval scaling
    
    rie = ReplaceImputeEncode(data_map=attribute_map, 
                              binary_encoding = encoding,
                              nominal_encoding= encoding, 
                              interval_scale = scale, 
                              drop=False, display=True)
    #features_map = rie.draft_features_map(df)
    stratefydf= df.groupby('clean_BinaryDefault', group_keys=False).apply(lambda x: x.sample(18000))
    encoded_df = rie.fit_transform(stratefydf)
    # columns =['ClosedMth','ClosedYr']
    # encoded_df.drop(index=columns, axis=1, inplace=True)
    
    y = encoded_df[target] # The target is not scaled or imputed
    X = encoded_df.drop([target] ,axis=1)

#'EstimatedLoss','EstimatedReturn','LenderYield','EstimatedEffectiveYield','CreationYr','ClosedMth','ClosedYr', 
#'EstimatedEffectiveYield','CreditPullYr', 'BorrowerRate', 'BorrowerAPR'
    X_train, X_validate, y_train, y_validate = \
            train_test_split(X, y, test_size = 0.4, random_state=12345, stratify=y)
    
    encoded_df.describe()         
# #######################################################################################            

if Run_Tree:
    print("\n*************************************************************")
    print("\nPart 3: Decision Tree loops for parameters optimization ")
    print("** 10-Fold Cross-Validation for Decision Tree Maximum Depth  **")
    print("** Hyperparameter Optimization  based  on  Maximum  F1-Score **")
    print("\n*************************************************************")

    score_list      = ['accuracy', 'precision', 'recall',  'f1']
    n_list          = len(score_list)
    bestd           = 0     #depth
    bestl           = 0     # Leaf
    best_size       = 0     #create a list for split size
    leafsize        = [ 7, 10]
    depthlist       = [5, 7, 10]
    splitsize       = [6, 10]  # 2*[3,5,7,10]
    best_score      = 0
    for m in splitsize:
        for l in leafsize:
            for d in depthlist:
                print("\nTree Depth: ", d,"  leaf size: ", l, "  Split_size: ", m)
            
                dtc = DecisionTreeClassifier(max_depth=d, min_samples_leaf=l, 
                                      min_samples_split=m,random_state=12345)
                dtc = dtc.fit(X, y)
                scores = cross_validate(dtc, X, y, scoring=score_list, cv=10,
                                return_train_score=False, n_jobs=-1)

                print("{:.<20s}{:>6s}{:>13s}".format("Metric","Mean", 
                                                      "Std. Dev."))
                for s in score_list:
                    var = "test_"+s
                    mean = scores[var].mean()
                    std  = scores[var].std()
                    print("{:.<20s}{:>7.4f}{:>10.4f}".format(s, mean, std))
                    if mean > best_score and s=='f1':
                        best_score          = mean
                        best_depthsize      = d
                        best_leafsize       = l
                        best_splitsize      = m

  
    print("\nEvaluating Using Entire Dataset")
    print("Best Depth =      ", best_depthsize)
    print("Best Leaf Size =  ", best_leafsize)
    print("Best Split Size = ", best_splitsize)

if DecisionTree_run:      
    print("****************************************************************************")
    print("\nDecision Tree with Best Depth= ", best_depthsize , "Best Leaf Size = ",best_leafsize ,"Best Split Size = ", best_splitsize)
    print("\n******************* Best Decision Tree 60/40 Validation ******************")
    print("****************************************************************************")

    dtc = DecisionTreeClassifier(max_depth=best_depthsize, min_samples_leaf=best_leafsize, 
                                      min_samples_split=best_splitsize,random_state=12345)
    dtc.fit(X_train, y_train)
    tree_classifier.display_importance(dtc, X_train.columns, top=35, plot=True)
    tree_classifier.display_split_metrics(dtc, X_train, y_train,X_validate, y_validate)


    
    
# #######################################################################
print("\n****************************************************************")
print("\n*********************** RANDOM FOREST **************************")
if Forest_CV:
    # Cross-Validation
    score_list      = ['accuracy', 'precision', 'recall',  'f1']
    n_list          = len(score_list)
    estimators_list = [6,8,9]
    best_size       = 2
    best_split_size = 2*best_size
    depth_list      = [ 6, 10,12]  #23
    features_list   = [ 15,20,24]
    best_score    = 0
    for e in estimators_list:
        for d in depth_list:
            for features in features_list:
                leaf_size  = round(X.shape[0]/600)  #changed from 1000
                split_size = 2*leaf_size
                print("\nNumber of Trees: ", e, "Max_Depth: ", d,
                      "Max Features: ", features)
                rfc = RandomForestClassifier(n_estimators=e, criterion="gini",
                            min_samples_split=split_size, max_depth=d,
                            min_samples_leaf=leaf_size, max_features=features, 
                            n_jobs=1, bootstrap=True, random_state=12345)
                scores = cross_validate(rfc, X, y, scoring=score_list, \
                                        return_train_score=False, cv=10)
                
                print("{:.<20s}{:>6s}{:>13s}".format("Metric","Mean", 
                                                     "Std. Dev."))
                for s in score_list:
                    var = "test_"+s
                    mean = scores[var].mean()
                    std  = scores[var].std()
                    print("{:.<20s}{:>7.4f}{:>10.4f}".format(s, mean, std))
                    if mean > best_score and s=='f1':
                        best_score      = mean
                        best_estimator  = e
                        best_depth      = d
                        best_features   = features
                        best_leaf_size  = leaf_size
                        best_split_size = split_size
                        
                        
if Forest_Eval:

    # Evaluate the random forest with the best configuration
    print("\n****************************************************************")
    print("\n******************** Best RANDOM FOREST ************************")
    print("\nEvaluating Using 60/40 Partition")
    
    print("Best Trees=", best_estimator)
    print("Best Depth=", best_depth)
    print("Best Leaf Size = ", best_leaf_size)
    print("Best Split Size = ", best_split_size)
    print("Best Max Features = ", best_features)
    rfc = RandomForestClassifier(n_estimators=e, criterion="gini", \
                            min_samples_split=best_split_size, 
                            max_depth=best_depth,
                            min_samples_leaf=best_leaf_size, 
                            max_features=best_features, 
                            n_jobs=-1, bootstrap=True, random_state=12345)
    rfc= rfc.fit(X_train, y_train)
    
    forest_classifier.display_split_metrics(rfc, X_train, y_train, \
                                            X_validate, y_validate)
    forest_classifier.display_importance(rfc, X.columns, plot=True)
    
    # shows Probability! 
    probability= rfc.predict_proba(X)
    print(probability)
    #print(y)
    
    print(sorted(zip(rfc.classes_, probability[0]), key=lambda x:x[1])[-80:])
    

    # save model to file
    pickle.dump(rfc, open("datamodeldeploy.dat", "wb"))
    
    # Load model from Pickle file
    loaded_modelRF = pickle.load(open("datamodeldeploy.dat", "rb"))
    
    #Predict from inputs 
    #List of Columns --> Index(['ListingCategory (numeric)', 'EmploymentStatusDuration',
       # 'CreditScoreRangeLower', 'CreditScoreRangeUpper',
       # 'TotalCreditLinespast7years', 'OpenRevolvingAccounts',
       # 'OpenRevolvingMonthlyPayment', 'TotalInquiries', 'AmountDelinquent',
       # 'RevolvingCreditBalance', 'AvailableBankcardCredit',
       # 'StatedMonthlyIncome', 'LoanOriginalAmount', 'LoanDateYr',
       # 'MonthlyLoanPayment', 'clean_IncomeRange', 'CreditGrade0:A',
       # 'CreditGrade1:AA', 'CreditGrade2:B', 'CreditGrade3:C', 'CreditGrade4:D',
       # 'CreditGrade5:E', 'CreditGrade6:HR', 'CreditGrade7:NC', 'Term12',
       # 'Term36', 'Term60']
    # Test features results-  Zero
    result= loaded_modelRF.predict([[1,	134,	660,	679,	21,	7,	219,	7,	0,	4115,	2069,	
                                     4417,	7000,	2011,	304.84,	53004,0,	0,	0,	1,	0,	0,	0,	
                                     0,	0,	1,	0 ]])

    print("Probability result equal", result )
    
    # Test features results-  One
    result= loaded_modelRF.predict([[0,	90.81775629,	672.1006099,	691.1006099,	26.22217245,0, 0,
                                     	6.926385209,	1164.349563,	16912.94719,	9775.625406,	4583,
                                         1000,	2006,	32.27,	54996,	0,	0,	0,	1,	0,	0,	0,	0,	0,	1,	0]])
    
   
    print("Probability result equal", result )
    



#############################################################################
