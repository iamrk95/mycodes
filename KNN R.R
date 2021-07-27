###>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> KNN <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<###
############################## 1ST QUESTION ###################################
#LOAD THE DATA PACKAGES
library(readr)
library(class)
library(caTools)
library(base)
library(ggplot2)
library(DescTools)

#LOAD THE DATASET
glass <- read.csv("glass.csv")

#CHECK FOR NA VALUES
sum(is.na(glass))

#CHECK FOR OULIERS
boxplot(glass)

#OUTLIERS TREATMENT IS DONE ON RI,Ca AND Ba VARIABLES AS IT HAS HIGH OUTLIERS
#RI
boxplot(glass$RI)
qunt1 <- quantile(glass$RI,probs = c(.25,.75),na.rm = T)
caps <- quantile(glass$RI, probs = c(.01,.99), na.rm = T)
H <- 1.5*IQR(glass$RI,na.rm = T)
glass$RI[glass$RI<(qunt1[1]-H)] <- caps[1]
glass$RI[glass$RI>(qunt1[2]+H)] <- caps[2]
boxplot(glass$RI)

#Ca
boxplot(glass$Ca)
qunt1 <- quantile(glass$Ca,probs = c(.25,.75),na.rm = T)
caps <- quantile(glass$Ca, probs = c(.01,.99), na.rm = T)
H <- 1.5*IQR(glass$Ca,na.rm = T)
glass$Ca[glass$Ca<(qunt1[1]-H)] <- caps[1]
glass$Ca[glass$Ca>(qunt1[2]+H)] <- caps[2]
boxplot(glass$Ca)

#Ba
boxplot(glass$Ba)
qunt1 <- quantile(glass$Ba,probs = c(.25,.75),na.rm = T)
caps <- quantile(glass$Ba, probs = c(.01,.99), na.rm = T)
H <- 1.5*IQR(glass$Ba,na.rm = T)
glass$Ba[glass$Ba<(qunt1[1]-H)] <- caps[1]
glass$Ba[glass$Ba>(qunt1[2]+H)] <- caps[2]
boxplot(glass$Ba)

#EDA
#TABLE OF TYPE
table(glass$Type)

#Normalize the data
norm <- function(x){
  return ((x-min(x))/(max(x)-min(x)))
}

glass_norm <- as.data.frame(lapply(glass[,c(1:9)], norm))
summary(glass_norm)

#CBIND THE STANDARDIZED DATA WITH ORIGINAL DATA
glass_data <- cbind(glass_norm,glass[10])
summary(glass_data)

#CREATING TRAIN AND TEST DATA
set.seed(101)
sample <- sample.split(glass_data$Type,SplitRatio = 0.80)
glass_train <- subset(glass_data,sample==TRUE)
glass_test <- subset(glass_data,sample==FALSE)

#KNN MODEL
glass_pred <- knn(glass_train[1:9],glass_test[1:9],glass_train$Type,k=12)

#EVALUATING THE PERFORMANCE OF THE MODEL
confusion_test <- table(x = glass_test$Type, y = glass_pred)
confusion_test

#ERROR IN PREDICTION
error <- mean(glass_pred!= glass_test$Type)
error

#ACCURACY OF MODEL
Accuracy <- sum(diag(confusion_test))/sum(confusion_test)
Accuracy*100


#COMPARING TEST ACCURACY WITH TRAIN ACCURACY
glass_train_pred <- knn(glass_train[1:9],glass_train[1:9],glass_train$Type,k=12)

confusion_train <- table(x = glass_train$Type, y = glass_train_pred)
confusion_train

#ERROR IN PREDICTION
error <- mean(glass_train_pred!= glass_train$Type)
error

#ACCURACY OF MODEL
Accuracy <- sum(diag(confusion_train))/sum(confusion_train)
Accuracy*100

#OPTIMUM K VALUE
#TEST DATA ACCURACY OF 28 K VALUES
i=1
k.opt = 1
for(i in 1:28){
  knn.mod <- knn(glass_train[1:9],glass_test[1:9],glass_train$Type,k=i)
  k.opt[i] <- 100*sum(glass_test$Type==knn.mod)/NROW(glass_test$Type)
  k=i
  cat(k,'=',k.opt[i],'
      ')
}

#TRAIN DATA ACUURACY OF 28 K VALUES
i=1
k.opt = 1
for(i in 1:28){
  knn.mod <- knn(glass_train[1:9],glass_train[1:9],glass_train$Type,k=i)
  k.opt[i] <- 100*sum(glass_train$Type==knn.mod)/NROW(glass_train$Type)
  k=i
  cat(k,'=',k.opt[i],'
      ')
}
###################################################################################