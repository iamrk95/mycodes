########################K-MEANS CLUSTERING#############################
#EastWestAirlines Dataset
#Load the packages
library(readxl)
library(data.table)
library(plyr)

#Load the dataset
airline_1 <- read_excel("EastWestAirlines.xlsx",sheet = "data")
colnames(airline_1)

#Remove the irrelevant data
airline <- airline_1[,-c(1,12)]
str(airline)

#Check For NA values
sum(is.na(airline))

#Check for duplicated values 
dup <- duplicated(airline)
sum(dup)  #Shows 1 duplicated value

#Remove the duplicated values
airline <- airline[!duplicated(airline), ]

#Check again for duplicated value
dup <- duplicated(airline)
sum(dup)

#check for outliers
boxplot(airline)

#winsorize to treat outliers in Balance variable
boxplot(airline$Balance)
y <- boxplot(airline$Balance)
length(y$out)

qunt1 <- quantile(airline$Balance,probs = c(.25,.75),na.rm = TRUE)
qunt1 # 25% 23092, # 75% = 51452
caps <- quantile(airline$Balance, probs = c(.01,.99), na.rm = T)
caps # 1 % = 18534.25, 99% = 459795.39
H <- 1.5*IQR( airline$Balance,na.rm = T)
H # 110806.1
airline$Balance[airline$Balance<(qunt1[1]-H)] <- caps[1]
airline$Balance[airline$Balance>(qunt1[2]+H)] <- caps[2]
boxplot(airline$Balance)

#winsorize to treat outliers in Bonus_miles variable
x <- boxplot(airline$Bonus_miles)
length(x$out)

qunt1 <- quantile(airline$Bonus_miles,probs = c(.25,.75),na.rm = TRUE)
qunt1 # 25% 1250.00, # 75% =23810.75
caps <- quantile(airline$Bonus_miles, probs = c(.01,.99), na.rm = T)
caps # 1 % = 0, 99% = 104194 
H <- 1.5*IQR( airline$Bonus_miles,na.rm = T)
H # 33841.12
airline$Bonus_miles[airline$Bonus_miles<(qunt1[1]-H)] <- caps[1]
airline$Bonus_miles[airline$Bonus_miles>(qunt1[2]+H)] <- caps[2]
boxplot(airline$Bonus_miles)

##winsorize to treat outliers in Qual_miles variable
boxplot(airline$Qual_miles)

qunt1 <- quantile(airline$Qual_miles,probs = c(.25,.75),na.rm = TRUE)
qunt1 # 25% 0, # 75% = 0
caps <- quantile(airline$Qual_miles, probs = c(.01,.99), na.rm = T)
caps # 1 % = 0, 99% = 4121.55
H <- 1.5*IQR( airline$Qual_miles,na.rm = T)
H # 0
airline$Qual_miles[airline$Qual_miles<(qunt1[1]-H)] <- caps[1]
airline$Qual_miles[airline$Qual_miles>(qunt1[2]+H)] <- caps[2]
boxplot(airline$Qual_miles)

#Normalize the data
norm <- function(x){
  return ((x-min(x))/(max(x)-min(x)))
}

airline_norm <- as.data.frame(lapply(airline, norm))
summary(airline_norm)

# Elbow curve to decide the k value
twss <- NULL
for (i in 2:8) {
  twss <- c(twss, kmeans(airline_norm, centers = i)$tot.withinss)
}
twss

# Look for an "elbow" in the scree plot
plot(2:8, twss, type = "b", xlab = "Number of Clusters", ylab = "Within groups sum of squares")
title(sub = "K-Means Clustering Scree-Plot")


# Cluster Solution , trying different k values
fit <- kmeans(airline_norm,2) #WITH 2 CLUSTERS
str(fit)

fit <- kmeans(airline_norm, 3)  #WITH 3 CLUSTERS
str(fit)

fit <- kmeans(airline_norm,4)    #WITH 4 CLUSTERS
str(fit)

# FROM THE ABOVE TRAIL ON DIFFERENT WE FIND THAT K=3 GIVES 
#BEST RESULT WITH LOW WITHINESS VALUE AND HIGH BETWEENESS VALUE

fit <- kmeans(airline_norm, 3)  #WITH 3 CLUSTERS
str(fit)

fit$cluster

airline_final <- data.frame(fit$cluster, airline) # Append cluster NUMBER

aggregate(airline[, 1:10], by = list(fit$cluster), FUN = mean)

#Split the data based on cluster groups
group1 <- subset(airline_final,fit.cluster =='1')
group2 <- subset(airline_final,fit.cluster =='2')
group3 <- subset(airline_final,fit.cluster =='3')

#EDA on clustered groups group1 group2 and group3
boxplot(airline_final)

#Balance
hist(group1$Balance)
hist(group2$Balance)
hist(group3$Balance)

sd(group1$Balance)
sd(group2$Balance)
sd(group3$Balance)

#Bonus_Miles
hist(group1$Bonus_miles)
hist(group2$Bonus_miles)
hist(group3$Bonus_miles)

sd(group1$Bonus_miles)
sd(group2$Bonus_miles)
sd(group3$Bonus_miles)

#Read the dataset into csv file
library(readr)
write_csv(airline_final, "hclustairline.csv")
getwd()
##############################################################################
