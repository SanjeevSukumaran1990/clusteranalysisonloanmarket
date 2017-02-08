
# coding: utf-8

# # CRISP-DM 
# 
# #### I will be using the CRISP-DM methodology to solve this problem.Hence this notebook will have be divided into following categories:
#    * Business Understanding
#    * Data Understanding
#    * Data Preparation
#    * Modelling
#    * Evaluation
# 

# ## Business Understanding

# **Change Financial** which is a small regional bank based in Washington, DC wants to enter into home loans market.The task is to understand with the help of **Home Mortage Disclosure Act dataset** that, whether its a smart move to enter to the home loans market or not?
# 

# ## Data Understanding

# The dataset is  Home Mortgage Disclosure Act dataset which is published annually by the federal govt and has the basic data on all home loan applications recieved by lenders across the US.
# 
# **Loans Data**:Contains the home loans originated within the states where Change Financial Operates.<br/>
# **Institution data**:Data about the originating institutions are submitted by institution themselves

# In[4]:

#import loan and well as institutions data

#First understand loans data
get_ipython().magic('matplotlib inline')
import matplotlib.pyplot as plt
plt.style.use('ggplot')
import pandas as pd
import numpy as np
loans_data=pd.read_csv('F:\\capital one\\data-challenge-data-master\\data-challenge-data-master\\2012_to_2014_loans_data\\2012_to_2014_loans_data.csv')
#loans_data.head(1)
len(loans_data)
#len(loans_data.columns)

#loans_data.plot()  ##looking into 
#loans_data.Loan_Amount_000.astype(int).describe()
loans_data.columns


# ### Below is the description of the variables for loan_data dataset:
# 
# The data set consist of 1321158 rows and 24 colomns.
# 
# Out of which

# ## Data preparation and Exploration for loans dataset

# As noticed some error messages are shown as some of the colomns have not been casted correctly hence converting the colomns into the correct type.When looking into the data some of the colomns have not been casted correctly,and it also causes pandas to process a bit slowly hence converting colomns to their correct type.
# Before this we have to check for null values and remove all the outliers.Lets look for null values first.
# First we have to replace 'NA  ' with na.
# **To decide on the numbers of bins first we have to check the distribution and check how many of the loan_amounts fall in each category.In the bucket 300 to 417 and 418 to 700 is to distinguish between Conforming and Jumbo loans.**

# In[30]:

bins = [0,100,200, 300,500,700,1000, 100000]
group_names = ['upto100', '100to200', '200to300','300to417','418to700','700to1000','1000to100000']
loans_data['LoanAmount_Cat']=pd.cut(loans_data.Loan_Amount_000, bins, labels=group_names)
pd.value_counts(loans_data.LoanAmount_Cat)




# # Data Mungling

# ## First  Merging the two data sets and adding buckets

# # declaring hmda_init

# In[9]:

def hmda_init():
    user_columns=['Respondent_ID','Respondent_Name_TS']
    institution_data=pd.read_csv('F:\\capital one\\data-challenge-data-master\\data-challenge-data-master\\2012_to_2014_institutions_data\\2012_to_2014_institutions_data.csv',usecols=user_columns)
    
    loans_data=pd.read_csv('F:\\capital one\\data-challenge-data-master\\data-challenge-data-master\\2012_to_2014_loans_data\\2012_to_2014_loans_data.csv')
    new=pd.merge(loans_data,institution_data,on='Respondent_ID',how='inner',suffixes=(' ',' '))
    new.rename({'Respondent_Name_TS':'Respondent_Name'})
    bins = [0,100,200, 300,500,700,1000, 100000]
    group_names = ['upto100', '100to200', '200to300','300to417','418to700','700to1000','1000to100000']
    new['LoanAmount_Cat']=pd.cut(new.Loan_Amount_000, bins, labels=group_names)
    #new.T.groupby(level=0).first().T
    #pd.value_counts(loans_data.LoanAmount_Cat)
    return(new)


# In[10]:

# after merging and bucket creation,data is then fed to dataframe 's'
s=hmda_init()
s.columns


# ## Important note:Throughout this work i will be using the newly created dataframe returned from hmda_init() function

# # defining hmda_to_json(data,states,conventional_conforming)

# Now before defining this functions here are the consideration for the function.<br/>
# 1.What will be the input format for filtering the state?<br/>
# Ans:Input format will be a string .For Example:Illinois will be represented as IL.**Please pass an array along with states name**
# 
# 
# 
# 2 In conventional_conforming please enter either 'Y' or 'N'
# 
# 3.Optionally i have included additional variable for years,**Input should be fed in the form of array**
# 
# 4.path

# In[4]:

def hmda_to_json(data,path=None,states=None,conventional_conforming=None,Year=None):
    if(states):
        data=data[data.State.isin(states)]
    else:
        pass
    if(conventional_conforming):
        if(conventional_conforming=='Y'):
            data=data[(data.Conventional_Status=='Conventional') & (data.Conforming_Status=='Conforming')]
        elif(conventional_conforming=='N'):
            data=data[(data.Conventional_Status!='Conventional') & (data.Conforming_Status!='Conforming')]
    if(Year):
        data=data[data.As_of_Year.isin(Year)]
    if(path):
        data.to_json(path)
    else:
        data.to_json('F:\\capital one\\data-challenge-data-master\\data-challenge-data-master\\2012_to_2014_institutions_data\\san.json')


# In[70]:

hmda_to_json(s,states=['DC'],conventional_conforming='Y')


# # Quality Check/Data Exploration

# Lets do an initial assessment of the quality of dataset
# Checking of null values

# ## Quality Assessment of Loan_Amount_000

# In[14]:

s.dtypes  ##integer type


# In[17]:

s.Loan_Amount_000.isnull().sum()  ##initial assessment shows that the number of null values are zero,need to check deeper


# In[18]:

s.Loan_Amount_000.describe() 


# **Mean**=287<br/>
# **Std**=1010.53<br/>
# **max**=98625<br/>
# **min**=1<br/>
# 
# As we can see from the initial range check,that Loan amount ranges from  1 to 98625.Mean and std is 287 and 1010.83 respectively.That suggests presence of outliers

# ### Further investigation of outliers

# In[33]:

s.Loan_Amount_000.plot(kind='box') //as the figure suggest lots of outliers in our dataset


# 

# ## Recommendations to improve the quality of Loan_Amount_000 variable

# * There are various ways to handle outlier
# One such way is deleting all the values which are less or more than 3 standard deviations away from the mean.
# Other is by the method of smooth binning ,i.e dividing the data into buckets and then smooth each buckets by mean of data or by boundaries

# * Depending upon the implementation we should perform transformation of data,since this dataset is rightskewed hence we can do the following approach<br/>
# 1.Binning of data into different categories<br/>
# 2.Logarithmic transformation<br/>

# * Check for any missing values.Depending upon the use case we can either delete or replace the missing values with mean or median with respect to the lender or the location

# # Quality Assessment for Respondent_Name

# In[25]:

len(s.groupby('Respondent_Name_TS')['Respondent_Name_TS'].count())


# In[24]:

s.Respondent_Name_TS.value_counts().head()


# In[29]:

s.Respondent_Name_TS.value_counts(sort=True,ascending=True).head()


# There are 1717 unique respondent Name.Another intresting observations is Wells Fargo Bank ,FA has the highest number of applications,followed by Quicken loans and Suntrust mortgage.United Bank,The citizens national bank,all have the least number of applications.

# ## Checking for null values

# In[31]:

s.Respondent_Name_TS.isnull().sum()  # initial assessment shows number of null values as zero


# ## Conclusion regarding quality of data for Respondent_Name variable

# * This particular variable is clean.But this variable can be used to derive the Lenders name in loans_data dataset by joining it 
#   with institution dataset.Hence we have to ensure that dataset is any datapoint is not missing in respondent_id variable as   well as respondent_name_ts variable.
# * Another quality check is to check for duplicated rows for the same lender being assigned different id's unintentionally.It might cause major confusion

# ## Other variables of potential importance

# Other variables which i consider which should be important are variables below,because these variables are really important for our analysis as per our business needs.<br/>
# 1.Conforming_Limit_000  <br/>
# 2.Conventional_Status<br/>
# 3.Conforming_Status<br/>
# 4.Applicant Income.<br/>
# 5.Lien_Status_Description <br/>
# 
# Because our business decision will be on the basis of how risky it is to enter into loan market.If its a low risk and high output,the company should enter into home loan market.Hence we have to consider all these variables into consideration.Hence these are the variable which needs proper quality assessment.
# 

# ## Other variables to look out for and final data preparation based on the observations.

# In[47]:

s.isnull().sum()


# Further investigation in the dataset found that all those colomns which had missing values for County_Name,Conforming_Limit_000 and Tract_to_MSA_MD_Income_Pct had same rows with NA values.

# In[48]:

s.isnull().sum()  ## again checking for null values


# Further investigation of the dataset found a major flaw which might go undetected.Some of the null values were represented by string 'NA  '.Hence convert all the 'NA  ' to np.nan and then decide what to do depending upon the importance of the variable.
# Hence next step would be to strip all the values of the rowset containing string of any spaces.

# In[50]:

s.replace('NA  ',np.nan,inplace=True)
s.isnull().sum()


# In[51]:

s.replace('NA   ',np.nan,inplace=True)
s.isnull().sum()


# This makes sense as well,since it means that 281474 are regions which lies in metropolitan areas.
# 

# In[63]:

s.dtypes[s.dtypes==object]


# Hence next step will be converting all these variables to string type and then stripping of any spaces in the  values.
# Next step after that would be to convert each of the variables to there appropriate types.For example:Applicant_Income_000 should be stored as integer rather than string.

# ### To summarize:

# * Further investigation in the dataset found that all those colomns which had missing values for County_Name,Conforming_Limit_000 and Tract_to_MSA_MD_Income_Pct had same rows with NA values.
# * Further investigation of the dataset also showed a major flaw which might go undetected.Some of the null values were represented by string 'NA '.Hence convert all the 'NA ' to np.nan and then decide what to do depending upon the importance of the variable. Hence next step would be to strip all the values of the rowset containing string of any spaces.
# * Hence next step will be to identify all the colomns having type as string.
# * After that depending upon the use case will impute by mean or median or remove the variables.<br/>Next some the variables like Agency code as mentioned as string.Hence next step will be to convert all these strings to there    appropriate types.
# 
# * After that depending upon the use case will impute by mean or median or remove the variables.
#     Then again checking the null values.
# * Check the skewness of all the variables.
# * I will be considering new variable which is a ratio of applicant income and debt amount
# 

# # Please check the documentation regarding the metadata and the quality checks for a summarized view.

# # Detailed data preparation has been done in clustering module along with explaination.

# In[ ]:




# ## Craft a visual narrative

# In[ ]:

To decide whether Change Financial can invest on this market or not,we can perform the following .


# **Hypothesis**
# Change Financial should enter into home loan market,since the market size is increasing yearly irrespective of region.
# 
# **Metric**:Metric to define market size will be the number of applications

# ### Overall market

# In[38]:

new=pd.crosstab(index=s.State,columns=[s.Conventional_Conforming_Flag,s.Conventional_Status,s.Conforming_Status])


# In[30]:

new.plot(kind="bar", 
                 figsize=(8,8),
                 stacked=False)


#                Fig:Statewise trend of different loan types
#                      * Conforming and Non-Conventional
#                      * Jumbo and Conventional
#                      * Jumbo and Non-Conventional
#                      * Conforming and Conventional
#                

# # Modifying data 

# ### hmda_to_json function was modified which will help me generate year wise data

# My next step will be to extract data year wise by modifying hmda_to_pandas function to return dataframe.Which inturn i will be plotting yearwise.

# In[19]:

def hmda_to_pandas(data,states=None,conventional_conforming=None,Year=None):
    if(states):
        data=data[data.State.isin(states)]
    else:
        pass
    if(conventional_conforming):
        if(conventional_conforming=='Y'):
            data=data[(data.Conventional_Status=='Conventional') & (data.Conforming_Status=='Conforming')]
        elif(conventional_conforming=='N'):
            data=data[(data.Conventional_Status!='Conventional') & (data.Conforming_Status!='Conforming')]
    if(Year):
        data=data[data.As_of_Year.isin(Year)]
            
    return data


# In[37]:

s.As_of_Year.value_counts() # So the years which are present in the datasets are 2012,2013 and 2014


# ## Statewise market for year 2012

# In[47]:

year1=hmda_to_pandas(s,Year=[2012])


# In[48]:

crosstab1=pd.crosstab(index=year1.State,columns=[year1.Conventional_Conforming_Flag,year1.Conventional_Status,year1.Conforming_Status])


# In[50]:

crosstab1.plot(kind="bar", 
                 figsize=(8,8),
                 stacked=False)


# ## 2013
# 

# In[56]:

year2=hmda_to_pandas(s,Year=[2013])


# In[57]:

crosstab2=pd.crosstab(index=year2.State,columns=[year2.Conventional_Conforming_Flag,year2.Conventional_Status,year2.Conforming_Status])


# In[58]:

crosstab2.plot(kind="bar", 
                 figsize=(8,8),
                 stacked=False)


# ## 2014

# In[59]:

year3=hmda_to_pandas(s,Year=[2014])
crosstab3=pd.crosstab(index=year3.State,columns=[year3.Conventional_Conforming_Flag,year3.Conventional_Status,year3.Conforming_Status])

crosstab3.plot(kind="bar", 
                 figsize=(8,8),
                 stacked=False)


# In[62]:

crosstab1.plot(kind="bar", 
                 figsize=(8,8),
                 stacked=False,title='2012')

crosstab2.plot(kind="bar", 
                 figsize=(8,8),
                 stacked=False,title='2013')

crosstab3.plot(kind="bar", 
                 figsize=(8,8),
                 stacked=False,title='2014')


#                              Fig:Yearwise trend of market size across states

# ## Verdict

# Our hypothesis has been disproved.A company can invest into a particular kind of product only when there is a constant increase in the demand of the product.But as we can see from the visualization starting from year 2012 to 2014.It has seen a downward trends in the number of application across all categories.Hence it will **not be a good stratergic move for Change Financials** to enter into home loans market **at this point of time**.

# ## Key Findings

# * The most prominent types of loans are:<br/>
#   1.Conventional and Conforming<br/>
#   2.Non-Conventional and Conforming<br/>
# 
# * Two states which prominently has most number of applications are Maryland and Virginia.
#   
# * Almost all the types of types of loans have shown a downward trends over the years including the states with most numbers of years.
# 
# * The worst effected year was year 2014.
# * Recommendation will be to wait and watch.Maybe this is a temporary downward trend due to multiple reasons.Try to enter the market is going great.
# 

# 

# 

# In[ ]:



