from tkinter import *
from tkinter.tix import *
from tkinter import ttk, filedialog
from tkinter.messagebox import showinfo, showwarning
from tkinter.tix import Balloon
from click import style

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
from sklearn.metrics                              import accuracy_score

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

class RandomForestDefaultLoanVerifier():
    def __init__(self, guiWin) :
        self.mainGui = guiWin # Main Frame
        self.mainGui.columnconfigure(0, weight=1)
        self.mainGui.rowconfigure(0, weight=1)
        self.mainGui.geometry("2200x800")
        self.mainGui.title('Random Forest Loan Default Evaluator')

        # Run Functioncs:
        
        self.create_frames()
        self.widget_styling()
        self.configure_widgets()

    def create_frames(self):

        #Primary Container Frame
        self.mainframe = ttk.Frame(self.mainGui
            ,padding=''
            ,style = 'entryFrame.TFrame')
        self.mainframe.grid(column=1, row=0, sticky=(N, W, S, E))

        #Feature Label and Entry Frame:
        self.features_frame = ttk.Frame(self.mainframe 
            ,padding = '5 5 5'
            ,relief='sunken'
            ,borderwidth=5
            ,style='')
        self.features_frame.grid(column=2, columnspan=2, row=1, rowspan=3,  padx=25)

        #Create and Metrics Approval Frame
        self.approval_and_metrics_frame = ttk.Frame(self.mainframe
            ,padding='7 7 7 7'
            ,borderwidth=5
            ,style = 'entryFrame.TFrame'
            ,relief='')
        self.approval_and_metrics_frame.grid(column = 4, columnspan=2, row = 1, rowspan=10, padx=20, sticky=(N, W, S, E))


        #              "Hidden" Scroll Frame:      #
        """"""""""""""""""""""""""""""""""""""""""""
        '   Hidden frame to scroll Entry Labels     '

        """

            To show all the Entry Labels you may have to make the GUI Window
            Larger by dragging the corner or specifying larger dimensions. 
        
        """
    
        # Canvas inside MainFrame - this case the features frame
        self.my_canvas = Canvas(self.mainGui) # needs to be in mainframe
        self.my_canvas.grid(column=0,  row=0, padx=40, sticky=(N, W, S, E))

        # Adding a scrollbar to Canvas. 
        my_scrollbar = ttk.Scrollbar(self.mainGui, orient=VERTICAL, command=self.my_canvas.yview)
        my_scrollbar.grid(column=2, row=0,sticky=(N,S))
 
        # Configure The Canvas
        self.my_canvas.configure(yscrollcommand=my_scrollbar.set)
        self.my_canvas.bind('<Configure>', lambda e: self.my_canvas.configure(scrollregion=self.my_canvas.bbox("all")))

        # Another Frame inside Canvas
        self.second_frame = ttk.Frame(self.my_canvas)
   
 
        #Add new frame to a window in canvas.
        self.my_canvas.create_window((0, 0), window=self.second_frame, anchor='ne')

    def widget_styling(self):

        # Styling used for the widgets. 
        self.style = ttk.Style()
        self.style.configure('GuiName.TLabel', font=('Verdana', 25), foreground='#283d3b')
        self.style.configure('TLabel', font=('source sans pro', 12, 'bold'), foreground='#772e25')
        self.style.configure('entryFrame.TFrame', background='#283d3b')
        self.style.configure('submit.TButton', font=('source sans pro', 12), background='#c44536', foreground='#283d3b')


    def configure_widgets(self):

        self.title_label = ttk.Label(self.mainGui
            ,text='Loan Default Probability Calculator'
            ,style='GuiName.TLabel').grid(column=1, row=1)


        #              Feature Labels:            #
        """"""""""""""""""""""""""""""""""""""""""""
        'The Top Features that are needed for the model' 
        

        #30 Features
        self.entry_label = ['Listing Category (numeric)'
            ,'Employment Status Duration'
            ,'Lower Credit Score Range'
            ,'Upper Credit Score Range'
            ,'Total Credit Lines past 7 years'
            ,'Open Revolving Accounts'
            ,'Open Revolving Monthly Payment'
            ,'Total Inquiries'
            ,'Amount Delinquent'
            ,'Revolving CreditBalance'
            ,'Available Bankcard Credit'
            ,'Stated Monthly Income' 
            ,'Loan Original Amount'
            ,'Loan Date Yr'
            ,'Monthly Loan Payment'
            ,'Clean Income Range'
            ,'Credit Grade'
            ,'Term']

        mssgs = ['Listing Category (numeric): The category of the listing that the borrower selected when posting their listing: \
                0 - Not Available, \
                1 - Debt Consolidation,  2 - Home Improvement, 3 - Business, 4 - Personal Loan, 5 - Student Use, 6 - Auto, 7- Other, 8 - Baby&Adoption, 9 - Boat, 10 - Cosmetic Procedure, 11 - Engagement Ring, 12 - Green Loans, 13 - Household Expenses, 14 - Large Purchases, 15 - Medical/Dental, 16 - Motorcycle, 17 - RV, 18 - Taxes, 19 - Vacation, 20 - Wedding Loans'
            ,'Employment Status Duration: Enter the length in months of the employment status at the time the listing was created.'
            ,'Lower Credit Score Range: Lower bound of borrower\'s credit score as provided by a consumer credit rating agency '
            ,'Upper Credit Score Range: Upeer bound of borrower\'s credit score as provided by a consumer credit rating agency'
            ,'Total Credit Lines past 7 years: Number of credit lines in the past seven years at the time the credit profile was pulled'
            ,'Open Revolving Accounts: Number of open revolving accounts at the time the credit profile was pulled.'
            ,'Open Revolving Monthly Payment: Monthly payment on revolving accounts at the time the credit profile was pulled.'
            ,'Inquiries Last 6 Months: Number of inquiries in the past six months at the time the credit profile was pulled.'
            ,'Total Inquiries: Total number of inquiries at the time the credit profile was pulled.'
            ,'Amount Delinquent: Dollars delinquent at the time the credit profile was pulled.'
            ,'Revolving Credit Balance: Dollars of revolving credit at the time the credit profile was pulled.'
            ,'Available Bankcard Credit: Total available credit via bank card at the time the credit profile was pulled.'
            ,'Stated Monthly Income: Monthly income the borrower stated at the time the listing was created.'
            ,'Loan Original Amount: Origination amount of the loan'
            ,'Loan Date Yr: Loan date extracted from today\'s date'
            ,'Monthly Loan Payment: Scheduled monthly loan payment.'
            ,'Clean Income Range: Yearly Income of the borrower '
            ,'Credit Grade: Credit rating  AA, A, B, C, D,E, HR'
            ,'Term: The length of the loan expressed in months 12,36 or 60']


        for i in range(len(self.entry_label)):
            self.ltip = Balloon(self.second_frame)
            labels = ttk.Label(self.second_frame
                ,padding='1 5 5 8'
                ,text=self.entry_label[i]
                ,style = 'TLabel')
            labels.grid(column = 1, row = i, stick=W)
            self.ltip.bind_widget(labels, balloonmsg =mssgs[i])


        self.entryStrings = [StringVar(),StringVar(),StringVar(),StringVar(),StringVar(),
            StringVar(),StringVar(),StringVar(),StringVar(),StringVar(),
            StringVar(),StringVar(),StringVar(),StringVar(),StringVar(),
            StringVar(),StringVar(),StringVar()]

        length = len(self.entryStrings)
        for i in range(length):
            self.entry_one = ttk.Entry(self.second_frame
                ,textvariable=self.entryStrings[i]
                ,style='TEntry'
                ,justify=CENTER
                ,font=('consolas', 14, 'bold'))
            self.entry_one.grid(column=2, row = i, sticky=E)


        #                 Buttons:                #
        """"""""""""""""""""""""""""""""""""""""""""
    
        self.submit_button = ttk.Button(self.features_frame
            ,text='Submit Data For Approval'
            ,style='submit.TButton'
            ,command=self.get_entry_data) # get_entry_data function. 
        self.submit_button.grid(column=1, columnspan=2, row=1, pady=15, sticky=(N, W, S, E))


        self.run_model = ttk.Button(self.features_frame
            ,text='Run Model'
            ,style='submit.TButton'
            ,command=lambda: self.run_rf_model()) # lambda b/c python was running code upon starting program.
        self.run_model.grid(column=1, columnspan=2, row=2, pady=15,sticky=(N, W, S, E))

        self.clear_text = ttk.Button(self.features_frame
            ,text='Clear All Data'
            ,style='submit.TButton'
            ,command=self.del_entry_and_text_data)
        self.clear_text.grid(column=1, columnspan=2, row=3 ,pady=15,  sticky=(N, W, S, E))


        #               Model Metrics:             #
        """"""""""""""""""""""""""""""""""""""""""""
        '      Show the Metrics of the Model       '

        # Print List here. 
        self.greetings = Text(self.approval_and_metrics_frame
            ,wrap='word'
            ,height=30
            ,bg='#edddd4'
            ,fg='#284b63'
            ,width = 99
            ,font=('Verdana', 14))
        self.greetings.grid(column=2, row=6, sticky=(N, W, S, E), )

        self.greetings.insert(1.0, f'\nGeneral Instructions:\n\n\
            For best experience please follow the instructions bellow:\n\n\
            1) Please enter values in the entry boxes. \n\
            2) If unsure of the required inforamtion hover over the label for details\n\
            3) Program will notify if wrong data is entered\n\
            4) After typing in all data click on "Submit Data For Approval" \n\
            5) If no error message appears click "Run Model"\n\
            6) Click "Clear all Data" to enter new values.  ')

    def get_entry_data(self):
        # Get the input data into a dictionary
        # Check that the proper items are the required datatype. 

        data_values_for_model = {'Listing Category (numeric)': [],
            'Employment Status Duration': [],
            'Lower Credit Score Range': [],
            'Upper Credit Score Range': [],
            'Total Credit Lines past 7 years': [],
            'Open Revolving Accounts': [],
            'Open Revolving Monthly Payment': [],
            'Total Inquiries': [],
            'Amount Delinquent': [],
            'Revolving CreditBalance': [],
            'Available Bankcard Credit': [],
            'Stated Monthly Income': [],
            'Loan Original Amount': [],
            'Loan Date Yr': [],
            'Monthly Loan Payment': [],
            'Clean Income Range': [],
        }

        # Dictionary will be updated with 1 for the Entry value.
        # One-Hot Encodign Data for.predict()
        credit_grade_dic = {
            'CreditGrade0:A': 0,
            'CreditGrade1:AA':0,
            'CreditGrade2:B': 0,
            'CreditGrade3:C': 0,
            'CreditGrade4:D': 0,
            'CreditGrade5:E': 0,
            'CreditGrade6:HR': 0,
            'CreditGrade7:NC': 0,
        }

        term_dic= {'Term12': 0,
            'Term36': 0,
            'Term60': 0
            }

        # Gotten Grade is on the 17th row.
        gottenGrade = self.entryStrings[16].get().upper()
        possible_grades = ['A', 'AA', 'B', 'C', 'D', 'E', 'HR', 'NC']  

        if gottenGrade in possible_grades:
            if gottenGrade == 'A':
                val_Grade = {'CreditGrade0:A': 1}
                credit_grade_dic.update(val_Grade)
            elif gottenGrade == 'AA':
                val_Grade = {'CreditGrade1:AA': 1}
                credit_grade_dic.update(val_Grade)
            elif gottenGrade == 'B':
                val_Grade = {'CreditGrade2:B': 1}
                credit_grade_dic.update(val_Grade)
            elif gottenGrade == 'C':
                val_Grade = {'CreditGrade3:C': 1}
                credit_grade_dic.update(val_Grade)
            elif gottenGrade == 'D':
                val_Grade = {'CreditGrade4:D': 1}
                credit_grade_dic.update(val_Grade)
            elif gottenGrade == 'E':
                val_Grade = {'CreditGrade5:E': 1}
                credit_grade_dic.update(val_Grade)
            elif gottenGrade ==  'HR':
                val_Grade = {'CreditGrade6:HR': 1}
                credit_grade_dic.update(val_Grade)
            elif gottenGrade == 'NC':
                val_Grade = {'CreditGrade7:NC': 1}
                credit_grade_dic.update(val_Grade)
        else:
            warn_msg = gottenGrade + ' Please Enter the correct Credit Grade'
            showwarning(title='Credit Grade'
                    ,message=warn_msg)
            self.entryStrings[16].set("NOT A CORRECT VALUE!")
    
        try:
            term = int(self.entryStrings[17].get())
            possible_terms = [12, 36, 60]

            if term in possible_terms:
                if term == 12:
                    val_Term = {'Term12': 1}
                    term_dic.update(val_Term)
                elif term == 36:
                    val_Term = {'Term36': 1}
                    term_dic.update(val_Term)
                elif term == 60:
                    val_Term = {'Term60': 1}
                    term_dic.update(val_Term)
            else:
                warn_msg = gottenGrade + 'Please enter 12, 36 or 60'
                showwarning(title='Term '
                        ,message=warn_msg)
                self.entryStrings[17].set("Wrong Values")
        except:
            warn_msg = gottenGrade + 'Please Enter the correct Credit Grade'
            showwarning(title='Credit Grade'
                    ,message=warn_msg)
            self.entryStrings[17].set("Please Review Value!") 


        # Subtract two because the other on-hot encoded values are above.
        # If I find more time I will add more logic for each category to
        # ensure each value is entered within its proper range.
        for i in range(len(self.entry_label)-2):
            try:
                value = self.entryStrings[i].get()
                data_values_for_model[self.entry_label[i]] = float(value)
            except:
                warn_msg = value + 'Not a recognized number by the Global Number Association'
                showwarning(title='Please Enter a Number'
                    ,message=warn_msg)
                self.entryStrings[i].set("Update Value!")



        # Updating Dictionary 
        data_values_for_model.update(credit_grade_dic)
        data_values_for_model.update(term_dic)
        
        # List of values which will be passed through predict function
        self.finalValues = [v for k, v in data_values_for_model.items()]
        print(' ')
        print(self.finalValues)
        print(' ')
        # print(self.finalValues[16:])
        self.greetings.insert(1.0, f'\n\n\n\n\n\n\
            Running Model with these values:\n\n {self.finalValues}\n\n\n')


    def del_entry_and_text_data(self):
        
        for var in self.entryStrings:
            var.set('')
        
        # Delete Text() boxes;
        self.metrics_output.delete(1.0, 'end')
        self.approval_label.delete(1.0, 'end')

    
    def run_rf_model(self):
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


        RIE_Non_Reg = False
        Run_Tree    = False 
        DecisionTree_run = False
        Forest_CV   = False
        Forest_Eval = False
        Forest_Pred = True

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
            

        if Forest_Pred:
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
            '''
            Some GUI use cases to test logic. 

            useCasesWithGui = [7.0, 22.000000, 700.0, 719.0, 11.0, 4.0, 75.0, 5.0, 0.000000,
             571.000000, 1429.000000, 1333.0, 2000.0, 2011.0, 80.17, 15996.0]
             

            # result= loaded_modelRF.predict([[1,	134,	660,	679,	21,	7,	219,	7,	0,	4115,	2069,	
            #                                 4417,	7000,	2011,	304.84,	53004,0,	0,	0,	1,	0,	0,	0,	
            #                                 0,	0,	1,	0 ]])
            
            # print("Probability result equal", result )
            
            # # Test features results-  One
            # result= loaded_modelRF.predict([[0,	90.81775629,	672.1006099,	691.1006099,	26.22217245,0, 0,
            #                                     6.926385209,	1164.349563,	16912.94719,	9775.625406,	4583,
            #                                     1000,	2006,	32.27,	54996,	0,	0,	0,	1,	0,	0,	0,	0,	0,	1,	0]])
            
            '''

        #############################################################################
        
            # running prediction 
            a = self.finalValues
            result1= loaded_modelRF.predict([a])
            proba_a = loaded_modelRF.predict_proba([a])
            print("Probability result equal", result1, proba_a[0][1] )
            print(' ')
            print(' ')



        #      Model Approved Text and Color:      #
        """"""""""""""""""""""""""""""""""""""""""""
        '      Logic to change color               '
        '      Based on model results              '

        default = 'white'

        self.approval_label = Text(self.approval_and_metrics_frame
            ,wrap='word'
            ,height=6
            ,bg=default
            ,font=('Verdana', 20)
            ,width=70)
        self.approval_label.grid(column=0, columnspan=3, row=1, rowspan=4)

        # https://stackoverflow.com/questions/47591967/changing-the-colour-of-text-automatically-inserted-into-tkinter-widget
        #self.result1 = None

        n=result1[0]
        if n == 0: # Zero = Loan Default
            self.approval_label.tag_config('DeniedRequest', foreground='black', background='red', justify='center')
           
            self.approval_label.insert(1.0, f'\n\nProbability of Default:\n{round((proba_a[0][0])*100,2)}%\n\n\n', 'DeniedRequest')
        elif n == 1: # One => They will repay loand
            'Aprove'
            self.approval_label.tag_config('ApprovedRequest', foreground='black', background='green', justify='center')
            self.approval_label.insert(1.0, f'\n\nProbability of Repayment:\n{round((proba_a[0][1])*100,2)}%\n\n\n', 'ApprovedRequest')
        elif n != 1 or n != 2:
            pass


root = Tk()
my_loanChecker = RandomForestDefaultLoanVerifier(root)
root.mainloop()
