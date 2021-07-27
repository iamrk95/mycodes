###>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> KNN <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<###
#LOAD THE PACKAGES
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from feature_engine.outliers import Winsorizer
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split

#LOAD THE DATASET
glass = pd.read_csv("C:/Users/Admin/Desktop/Datasets/glass.csv")

#CHECK FOR NA AND DATATYPE
glass.info()

#CHECK FOR OUTLIERS
plt.boxplot(glass)

#OUTLIERS TREATMENT IS DONE ON RI,Ca AND Ba VARIABLES AS IT HAS HIGH OUTLIERS
#RI
plt.boxplot(glass.RI)
windsoriser = Winsorizer(capping_method='gaussian', 
                          tail='both', # cap left, right or both tails 
                          fold=1.5,
                          variables=['RI'])
glass['RI'] = windsoriser.fit_transform(glass[['RI']])
plt.boxplot(glass.RI)

#Ca
plt.boxplot(glass.Ca)
windsoriser = Winsorizer(capping_method='gaussian', 
                          tail='both', # cap left, right or both tails 
                          fold=1.5,
                          variables=['Ca'])
glass['Ca'] = windsoriser.fit_transform(glass[['Ca']])
plt.boxplot(glass.Ca)

#Ba
plt.boxplot(glass.Ba)
windsoriser = Winsorizer(capping_method='gaussian', 
                          tail='both', # cap left, right or both tails 
                          fold=1.5,
                          variables=['Ba'])
glass['Ba'] = windsoriser.fit_transform(glass[['Ba']])
plt.boxplot(glass.Ba)

#NORMALIZATION
def norm_func(i):
    x = (i-i.min())	/ (i.max()-i.min())
    return (x)

glass_n = norm_func(glass.iloc[:,0:9])
glass_n.describe()

#TRAIN AND TEST DATA SPLIT
X = np.array(glass_n.iloc[:,:])
Y = np.array(glass['Type'])

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2)

knn = KNeighborsClassifier(n_neighbors =12)
knn.fit(X_train, Y_train)

pred = knn.predict(X_test)
pred

# EVALUATE THE MODEL
print(accuracy_score(Y_test, pred))
pd.crosstab(Y_test, pred, rownames = ['Actual'], colnames= ['Predictions']) 


# ERROR IN TRAIN DATA
pred_train = knn.predict(X_train)
print(accuracy_score(Y_train, pred_train))
pd.crosstab(Y_train, pred_train, rownames=['Actual'], colnames = ['Predictions']) 

# CREATING THE EMPTY LIST VARIABLE 
acc = []

#OPTIMUM K VALUE

for i in range(2,50,2):
    neigh = KNeighborsClassifier(n_neighbors=i)
    neigh.fit(X_train, Y_train)
    train_acc = np.mean(neigh.predict(X_train) == Y_train)
    test_acc = np.mean(neigh.predict(X_test) == Y_test)
    acc.append([train_acc, test_acc])

# train accuracy plot 
plt.plot(np.arange(2,50,2),[i[0] for i in acc],"ro-")

# test accuracy plot
plt.plot(np.arange(2,50,2),[i[1] for i in acc],"bo-")
#############################################################################

