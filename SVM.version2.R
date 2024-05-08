# IMPORTING THE DATA
df <- read.csv("https://github.com/ahaywasUSD/576-Data-Group-Project/raw/main/data_synthetic_ATH.csv")
df$Risk.Profile <-as.factor(df$Risk.Profile) #Convert to factor

summary(df$Risk.Profile)  
#CONVERT TO FACTORS
  
# LOAD THE LIBRARIES
library(rsample) #FOR initial_split() STRATIFIED RANDOM SAMPLING
library(e1071) #SVM LIBRARY
library(caret)

##PARTITIONING THE DATA##
# CREATE INDICES FOR DATA PARTITIONING
set.seed(123)
train_index<- createDataPartition(df$Risk.Profile, p = 0.7, list = FALSE, times = 1)
valid_test_index <- setdiff(1:nrow(df), train_index)
valid_index <- sample(valid_test_index, size = length(valid_test_index) / 2)
test_index <- setdiff(valid_test_index, valid_index)

# CREATE PARTITIONS
train_set <- df[train_index, ]
valid_set <- df[valid_index, ]
test_set <- df[test_index, ]

# SAVE PARTITIONS AS CSV
write.csv(train_set, "train_set.csv", row.names = FALSE)
write.csv(valid_set, "valid_set.csv", row.names = FALSE)
write.csv(test_set, "test_set.csv", row.names = FALSE)

#BUILD SVM CLASSIFIER

kern_type<-"radial" #SPECIFY KERNEL TYPE

SVM_Model<- svm(Risk.Profile ~ Premium.Amount+Age+Credit.Score, 
                data = train_set, 
                type = "C-classification", #set to "eps-regression" for numeric prediction (this function does BOTH regresssion and classification problems)
                kernel = kern_type,
                cost= 10,                   #REGULARIZATION PARAMETER (degree of regularization lamba)
                gamma = 1/(ncol(training)-1), #DEFAULT KERNEL PARAMETER
                coef0 = 0,                    #DEFAULT KERNEL PARAMETER
                degree=2,                     #POLYNOMIAL KERNEL PARAMETER
                scale = FALSE)               
print(SVM_Model)

#REPORT IN AND OUT-OF-SAMPLE ERRORS (1-ACCURACY)
Accuracy_in_pretune <- mean(predict(SVM_Model, train_set)==train_set$Risk.Profile)
Accuracy_out_pretune <- mean(predict(SVM_Model, test_set)==test_set$Risk.Profile)

#TUNING THE SVM 
tune_control<-tune.control(cross=10) #SET K-FOLD CV PARAMETERS
set.seed(123)
TUNE <- tune.svm(x = train_set[,c(1,10,15)],
                 y = train_set[,"Risk.Profile"],
                 type = "C-classification",
                 kernel = kern_type,
                 tunecontrol=tune_control,
                 cost=c(0.01, 0.1, 1, 10, 100, 1000, 5000), #REGULARIZATION PARAMETER
                 gamma = 1/(ncol(train_set)-1), #KERNEL PARAMETER
                 coef0 = 0,           #KERNEL PARAMETER
                 degree = 2)
print(TUNE)

#RE-BUILD MODEL USING OPTIMAL TUNING PARAMETERS
SVM_Retune<- svm(Risk.Profile ~ Premium.Amount+Age+Credit.Score, 
                 data = train_set, 
                 type = "C-classification", 
                 kernel = kern_type,
                 degree = 2,
                 gamma = 0.0625,
                 coef0 = 0,
                 cost = 1000,
                 scale = FALSE)
print(SVM_Retune)

#REPORT IN AND OUT-OF-SAMPLE ERRORS (1-ACCURACY) ON RETUNED MODEL
Accuracy_in_tune<-mean(predict(SVM_Retune, train_set)==train_set$Risk.Profile)
Accuracy_out_tune<-mean(predict(SVM_Retune, test_set)==test_set$Risk.Profile)

#SUMMARIZE RESULTS IN A TABLE:
TUNE_TABLE <- matrix(c(Accuracy_in_pretune, 
                       Accuracy_in_tune,
                       Accuracy_out_pretune,
                       Accuracy_out_tune),
                     ncol=2, 
                     byrow=TRUE)

colnames(TUNE_TABLE) <- c('UNTUNED', 'TUNED')
rownames(TUNE_TABLE) <- c('ACCURACY_IN', 'ACCURACY_OUT')
TUNE_TABLE
