#############################K-MEANS CLUSTERING###############################
#EastWestAirlines Datset
#Load the packages
import pandas as pd
import matplotlib.pylab as plt
import seaborn as sns
import numpy as np
from sklearn.cluster import	KMeans

#Load the Dataset
airline1 = pd.read_excel("C:/Users/Admin/Desktop/Datasets/EastWestAirlines.xlsx",sheet_name="data")

#Summary of the Data
airline1.describe()
airline1.info()

#Remove the Irrelevant Data
airline = airline1.drop(["ID#","Award?"], axis=1)

#Check For NA Values
airline.isna().sum()

#Identify duplicates records in the data
duplicate = airline.duplicated()
sum(duplicate)

#Removing Duplicates
airline = airline.drop_duplicates() 
duplicate = airline.duplicated()
sum(duplicate)

# FIND THE OUTLIERS IN THE DATA
sns.boxplot(airline.Balance);plt.title('Boxplot');plt.show()

# Winsorization method is used on Balance column
from feature_engine.outliers import Winsorizer
windsoriser = Winsorizer(capping_method='gaussian', 
                          tail='both', # cap left, right or both tails 
                          fold=1.5,
                          variables=['Balance'])
airline['Balance'] = windsoriser.fit_transform(airline[['Balance']])
sns.boxplot(airline.Balance);plt.title('Boxplot');plt.show()

#Treat the outliers in Bonus_miles
sns.boxplot(airline.Bonus_miles);plt.title('Boxplot');plt.show()

# Winsorization method is used on Bonus_miles column
windsoriser = Winsorizer(capping_method='gaussian', 
                          tail='both', # cap left, right or both tails 
                          fold=1.5,
                          variables=['Bonus_miles'])
airline['Bonus_miles'] = windsoriser.fit_transform(airline[['Bonus_miles']])
sns.boxplot(airline.Bonus_miles);plt.title('Boxplot');plt.show()

#Treat the outliers in Qual_miles
sns.boxplot(airline.Qual_miles);plt.title('Boxplot');plt.show()

# Winsorization method is used on Qual_miles column
windsoriser = Winsorizer(capping_method='gaussian', 
                          tail='both', # cap left, right or both tails 
                          fold=1.5,
                          variables=['Qual_miles'])
airline['Qual_miles'] = windsoriser.fit_transform(airline[['Qual_miles']])
sns.boxplot(airline.Qual_miles);plt.title('Boxplot');plt.show()

# Normalization function 
def norm_func(i):
    x = (i-i.min())	/ (i.max()-i.min())
    return (x)

# Normalized data frame (considering the numerical part of data)
airline_norm = norm_func(airline.iloc[:, 0:])
airline_norm.describe()

###### scree plot or elbow curve ############
TWSS = []
k = list(range(2, 9))

for i in k:
    kmeans = KMeans(n_clusters = i)
    kmeans.fit(airline_norm)
    TWSS.append(kmeans.inertia_)
    
TWSS
# Scree plot 
plt.plot(k, TWSS, 'ro-');plt.xlabel("No_of_Clusters");plt.ylabel("total_within_SS")

# Selecting 3 clusters from the above scree plot which is the optimum number of clusters 
model = KMeans(n_clusters = 3)
model.fit(airline_norm)

model.labels_ # getting the labels of clusters assigned to each row 
mb = pd.Series(model.labels_)  # converting numpy array into pandas series object 
airline['clust'] = mb # creating a  new column and assigning it to new column 

airline.head()

#Rearranging the columns
airline = airline.iloc[:,[10,0,1,2,3,4,5,6,7,8,9]]
airline.head()

airline.iloc[:, 1:].groupby(airline.clust).mean()

#Split the data based on cluster group
group3 = airline[airline.clust == 0]
group2 = airline[airline.clust == 1]
group1 = airline[airline.clust == 2]

group1.describe()
group2.describe()
group3.describe()

#EDA on clustered groups group1 group2 and group3
plt.hist(group1.Balance)
plt.hist(group2.Balance)
plt.hist(group3.Balance)

plt.hist(group1.Bonus_miles)
plt.hist(group2.Bonus_miles)
plt.hist(group3.Bonus_miles)

# creating a csv file 
airline.to_csv("airlinecsvfile.csv", encoding = "utf-8")
import os
os.getcwd()

##############################################################################
