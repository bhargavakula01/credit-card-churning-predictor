import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

#Data preprocessing
cc_dataframe = pd.read_csv('BankChurners.csv')

cc_dataframe.loc[cc_dataframe['Marital_Status'] == 'Unknown', 'Marital_Status'] = 'Single'
cc_dataframe.loc[cc_dataframe['Education_Level'] == 'Unknown', 'Education_Level'] = 'Uneducated'
del cc_dataframe["Naive_Bayes_Classifier_Attrition_Flag_Card_Category_Contacts_Count_12_mon_Dependent_count_Education_Level_Months_Inactive_12_mon_2"]
del cc_dataframe["Naive_Bayes_Classifier_Attrition_Flag_Card_Category_Contacts_Count_12_mon_Dependent_count_Education_Level_Months_Inactive_12_mon_1"]

print(cc_dataframe.head(10))
print(cc_dataframe.info())

#Exploratory Data Analysis

#Plot 1: relationship between the education level and the credit limit
sns.catplot(x= 'Education_Level', y= 'Total_Revolving_Bal', data= cc_dataframe, kind= 'bar',)


#Plot 2: 



