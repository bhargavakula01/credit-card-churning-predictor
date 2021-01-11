import pandas as pd


cc_dataframe = pd.read_csv('BankChurners.csv')
#deleting last 2 columns
del cc_dataframe["Naive_Bayes_Classifier_Attrition_Flag_Card_Category_Contacts_Count_12_mon_Dependent_count_Education_Level_Months_Inactive_12_mon_2"]
del cc_dataframe["Naive_Bayes_Classifier_Attrition_Flag_Card_Category_Contacts_Count_12_mon_Dependent_count_Education_Level_Months_Inactive_12_mon_1"]

#Data preprocessing
print(cc_dataframe.head(10))
print(cc_dataframe.info())
cc_dataframe.loc[cc_dataframe['Marital_Status'] == 'Unknown', 'Marital_Status'] = 'Single'

cc_dataframe.loc[cc_dataframe['Education_Level'] == 'Unknown', 'Education_Level'] = 'Uneducated'
cc_dataframe.replace(['Uneducated','High School','College','Graduate','Post-Graduate','Doctorate'], [0, 1, 2, 3, 4, 5], inplace= True)
cc_dataframe.replace(['M', 'F'], [0, 1], inplace= True)
cc_dataframe.replace(['Married','Single','Divorced'], [0, 1, 2], inplace= True)
cc_dataframe.replace(['$60K - $80K','Less than $40K','$80K - $120K','$40K - $60K','$120K +','Unknown'], [70000, 40000, 100000, 50000,120000,76000], inplace= True)
cc_dataframe.replace(['Existing Customer','Attrited Customer'], [1, 0], inplace = True)

print(cc_dataframe.info())

y_var = cc_dataframe['Attrition_Flag']
x_var = cc_dataframe.drop(['Attrition_Flag','Card_Category'], axis= 1)


#Training the model
import pickle
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x_var, y_var, test_size=0.33)
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

randf = RandomForestClassifier()

randf.fit(x_train, y_train)

predictions_y = randf.predict(x_test)

score = round(accuracy_score(predictions_y, y_test) *100, 2)
print(score)

