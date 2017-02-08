
# coding: utf-8

# Hypothesis:<br/>
# A geographical location with high income,bigger loans,higher number of applicants are ideal for entering into home loan market.

# But Before proceeding with clustering we have to perform a lot of data cleaning and data preparation task.

# ## Data Cleaning

# I will be considering the data set obtained by combining loan data and institutions data for this purpose

# In[18]:

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
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


# In[19]:

c=hmda_init()
len(c)


# In[3]:

c.head()


# Checking the number of null values

# In[20]:

c.isnull().sum()


# ### But as we know from the data quality assessment stage,there are lots of 'NA  ' string throughout the data,which makes checking null values tricky

# Identify all the columns with all the colomn type as object

# In[21]:

c.dtypes[c.dtypes==object]


# Convert all the 'NA  ' to 'NA' by stripping spaces

# In[22]:

c.Census_Tract_Number=c.Census_Tract_Number.str.strip()
c.FFIEC_Median_Family_Income=c.FFIEC_Median_Family_Income=c.FFIEC_Median_Family_Income.str.strip()
c.Number_of_Owner_Occupied_Units=c.Number_of_Owner_Occupied_Units.str.strip()
c.Tract_to_MSA_MD_Income_Pct=c.Tract_to_MSA_MD_Income_Pct.str.strip()
c.Applicant_Income_000=c.Applicant_Income_000.str.strip()
c.County_Code=c.County_Code.str.strip()                           
c.MSA_MD=c.MSA_MD.str.strip().str.strip()                               
c.Respondent_ID=c.Respondent_ID.str.strip()                     
c.Tract_to_MSA_MD_Income_Pct=c.Tract_to_MSA_MD_Income_Pct.str.strip()      
c.MSA_MD_Description=c.MSA_MD_Description.str.strip()                
c.Loan_Purpose_Description=c.Loan_Purpose_Description.str.strip()         
c.Agency_Code_Description=c.Agency_Code_Description.str.strip()           
c.Lien_Status_Description=c.Lien_Status_Description.str.strip()          
c.Loan_Type_Description=c.Loan_Type_Description.str.strip()             
c.State=c.State.str.strip()                             
c.County_Name=c.County_Name.str.strip()                       
c.Conventional_Status=c.Conventional_Status.str.strip()              
c.Conforming_Status=c.Conforming_Status.str.strip()                 
c.Conventional_Conforming_Flag=c.Conventional_Conforming_Flag.str.strip()      
c.Respondent_Name_TS=c.Respondent_Name_TS.str.strip()     


# In[ ]:




# In[ ]:




# Next step will be to replace all the 'NA' to NaN

# In[23]:

c.replace('NA',np.nan,inplace=True)


# In[8]:

c.isnull().sum()


# Two of the columns which had large number of null values:<br/>
# i.County_Code and MSA_MD was deleted.Due to two reasons:<br/>
# 1.Not consideration of these variables in the cluster analysis.<br/>
# 2.Huge number of null values,rendering inablity to include these columns<br/>

# In[24]:

c.drop('County_Code',1,inplace=True)


# In[25]:

c.drop('MSA_MD',1,inplace=True)


# In[26]:

c.isnull().sum()


# **I will be removing all the null values which are in Census_Tract_Number since iam aggregating the data by this particular columns,to achieve the lowest number of granularity.**

# In[27]:

c.dropna(inplace=True)


# In[28]:

c.isnull().sum()


# Now we are having a clean data with all the null values removed

# Now converting all the variable to there appropriate type as follows:
# * Application_Income_000->int
# * Census_Tract_Number->int
# * FFIEC_Median_Family_Income->int
# * Number_of_Owner_Occupied_Units->int
# * Respondent_ID->int
# * Sequence_Number->int
# * Tract_to_MSA_MD_Income_Pct->float
# * Loan_Purpose_Description->string
# * Agency_Code_Description->string
# * Lien_Status_Description->string
# * Loan_Type_Description->string
# * State->string
# * County_Name->string
# * Conforming_Limit_000->int
# * Conventional_Status->string
# * Conforming_Status->string
# * Conventional_Conforming_Flag->string

# In[29]:

c.Applicant_Income_000=c.Applicant_Income_000.astype(int)
c.Census_Tract_Number=c.Census_Tract_Number.astype(float)
c.FFIEC_Median_Family_Income=c.Census_Tract_Number.astype(float)
c.Number_of_Owner_Occupied_Units=c.Number_of_Owner_Occupied_Units.astype(int)
#c.Respondent_ID=c.Respondent_ID.astype(int)
c.Sequence_Number=c.Sequence_Number.astype(int)
c.Tract_to_MSA_MD_Income_Pct=c.Tract_to_MSA_MD_Income_Pct.astype(float)
c.Conforming_Limit_000=c.Conforming_Limit_000.astype(int)


# In[15]:

c.dtypes


# # Data Manipulation

# This involves creating new variables and later grouping them by Census_Tract_Number

# Variable to be created:<br/>
# 1.% of people with first lien<br/>
# 2.% of people with second lien loan<br/>
# 
# **Reason**:Before investing into any venture the company should think about the risk involved.There might be geographical location where more number of people opt for second lien loan,maybe due to economic background or other reasons as well.
# In case of second lien,the company is at greater risk for loosing money.
# 

# In[32]:

df_new=pd.get_dummies(c.Lien_Status_Description)


# In[ ]:




# In[34]:

c= pd.concat([c, df_new], axis=1)


# Next column to be created is the one which identifies what percentage of the population has conventional loan and what percent does not.Conventional loan has greater possiblity to cause loss to the company than non-conventional loans

# In[35]:

df_new_1=pd.get_dummies(c.Conventional_Status)
df_new_1


# In[36]:

c= pd.concat([c, df_new_1], axis=1)


# # Creating one more variable which signifies applicant income to loan ratio

# In[37]:

c['IncometoLoan']=(c.Applicant_Income_000)/(c.Loan_Amount_000)


# ## Next we will select columns which will be of importance to us and copy it in a new dataframe

# In[38]:

new=c[['Census_Tract_Number','IncometoLoan','Tract_to_MSA_MD_Income_Pct','Conventional','Non-Conventional',
       'First Lien','Subordinate Lien','Number_of_Owner_Occupied_Units']]


# In[39]:

new.head()


# In[ ]:




# In[40]:

df=c.groupby('Census_Tract_Number').agg({'IncometoLoan':{'Incometoloan_new':'mean'},'Conventional':{'TotalNumber':'count','Conventional%':'sum'}
                                         ,'Non-Conventional':{'Non-Conventional%':'sum'},
                                      'Number_of_Owner_Occupied_Units':{'Number_of_Owner_Occupied_Units':'mean'},
                                      'Tract_to_MSA_MD_Income_Pct':{'Tract_to_MSA_MD_Income_Pct':'mean'},'First Lien':{'First Lien%':'sum'}})
                                    


# In[41]:

df.columns=df.columns.droplevel()


# In[42]:

df['Conventional%']=(df['Conventional%'])/(df['TotalNumber'])


# In[43]:

df['First Lien%']=(df['First Lien%'])/(df['TotalNumber'])


# In[44]:

df


# # New dataframe created for Cluster analysis

# **Below is the metadata of the newly formed dataset**

# The selection of the variable have been done on the basis of the available data.
# These variables are selected on the basis of strategy of lesser risk and better divident.Variable selection might vary as per difference in the strategy of the company.
# * Conventional%:Since conventional loans are those which are not govt sponsored,hence carry more risk,but better divident
# * IncometoLoan:Does the applicant have sufficient fund to pay back the loan.
# * FirstLien%:Since SecondLien% have higher risk.
# * Tract_to_MSA_MD_Income_Pct:Economic condition of the place.
# * IncometoLoan_new:Applicant income to loan amount(for better chances of repayment)
# * Number of Owner occupied Units:To get an insight into the real estate market of the region.
# * Total number of applicants

# In[45]:

df.describe()


# # Data Normalization

# In[73]:

from sklearn import preprocessing
normalizeddf=preprocessing.normalize(df)


# In[74]:

normalizeddf


# Deciding on the number of clusters

# In[77]:

from scipy import cluster

initial = [cluster.vq.kmeans(normalizeddf,i) for i in range(1,10)]
plt.plot([var for (cent,var) in initial])
plt.show()


# As per the figure the number of clusters can be 2,Hence going with 3

# In[135]:

get_ipython().magic('matplotlib inline')
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import pylab as pl
pca = PCA(n_components=2).fit(normalizeddf)
pca_2d = pca.transform(normalizeddf)
pl.figure('Reference Plot')
#pl.scatter(pca_2d[:, 0], pca_2d[:, 1], c=iris.target)
kmeans = KMeans(n_clusters=2, random_state=111)
kmeans.fit(normalizeddf)
pl.figure('K-means with 3 clusters')
pl.scatter(pca_2d[:, 0], pca_2d[:, 1], c=kmeans.labels_)
pl.show()



# In[136]:

(kmeans.cluster_centers_)


# ## Updated dataframe along with clusters assigned

# In[154]:

s=pd.DataFrame(kmeans.labels_)
s
s.index=df.index
s
pd.concat([df,s],axis=1)


# In[155]:

s=normalizeddf.mean(0)
s


# In[156]:

normalizeddf


# ## Cluster centroids for each clusters in each columns

# In[157]:

s=(kmeans.cluster_centers_)
s=pd.DataFrame(s)
s.columns=df.columns
s


# ## Actual mean across different columns

# In[158]:

s1=normalizeddf.mean(0)
s1=pd.DataFrame(s1)
s1.T


# ## What does each cluster mean

# **Cluster1:**Lower Incometoloan ratio,Number of people with first lien lower,Lower income,but high number of applications,greater number of non-conventional loans,higher number of applications<br/>
# **Cluster2:**Higher IncometoLoan ratio,Greater number of owner occupied units,high number of people with first lien,but lower number of applications<br/>

# As per me it would be stratergically better to identify census_tracts which have people with lower incometoloan ratio,higher number of applications,and more number of non-conventional loans.Hence we can enter into market corresponding to census_tracts in **Cluster1**

# In[ ]:



