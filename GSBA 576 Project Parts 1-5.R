#LOAD LIBRARIES
library(dplyr)
library(ggplot2)
library(reshape2)
library(caret)
library(lmridge) #for Ridge Regression
library(broom)
library(MASS)
library(lmridge) #for regularization
library(broom) #FOR glance() AND tidy()
library(Metrics) #FOR rmse()
library(e1071) #SVM LIBRARY
library(rsample) #FOR initial_split() STRATIFIED RANDOM SAMPLING
library(tidymodels) #INCLUDES parsnip PACKAGE FOR decision_tree()
library(caret) #FOR confusionMatrix()
library(rpart.plot)
library(tidymodels)
library(baguette) #FOR BAGGED TREES
library(xgboost) #FOR GRADIENT BOOSTING
library(caret) #FOR confusionMatrix()
library(vip) #FOR VARIABLE IMPORTANCE

#IMPORT THE DATA
df <- read.csv('https://raw.githubusercontent.com/ahaywasUSD/576-Data-Group-Project/main/data_synthetic_ATH.csv')
#CONVERT TO FACTORS
df <- df %>%
  mutate_if(is.character, as.factor)
summary(df)



#Nonlinear Transformations of Previous.Claims.History
df$Previous.Claims.History2 <- df$Previous.Claims.History^2
#df$Previous.Claims.History3 <- df$Previous.Claims.History^3
#df$Previous.Claims.History4 <- df$Previous.Claims.History^4
#df$Previous.Claims.History_LN <- log(df$Previous.Claims.History)
#summary(df)

#I.1 - Data Partitioning
 set.seed(123)
 #FRACTION OF DATA TO BE USED AS IN-SAMPLE TRAINING DATA
 p<-.7 #70% FOR TRAINING (IN-SAMPLE)
 #h<-.3 #HOLDOUT (FOR TESTING OUT-OF-SAMPLE)
 
 obs_count <- dim(df)[1] #TOTAL OBSERVATIONS IN DATA
 
 #OF OBSERVATIONS IN THE TRAINING DATA (IN-SAMPLE DATA)
 #floor() ROUNDS DOWN TO THE NEAREST WHOLE NUMBER
 training_size <- floor(p * obs_count)
 training_size
 #RANDOMLY SHUFFLES THE ROW NUMBERS OF ORIGINAL DATASET
 train_ind <- sample(obs_count, size = training_size)
 Training <- df[train_ind, ] #PULLS RANDOM ROWS FOR TRAINING
 Holdout <- df[-train_ind, ] #PULLS RANDOM ROWS FOR TESTING. this is the holdout data.
 
 #Randomly splitting Holdout into 2 (for Testing & Validation; 15% each)
 set.seed(123)
 obs_count_hold <- dim(Holdout)[1] 
 hold_ind <- sample(obs_count_hold, size = floor(0.5*obs_count_hold)) #0.5 weight for 50/50 split of 30% 

 Testing <- Holdout[hold_ind,]
 Validation <- Holdout[-hold_ind,]
 
 
 #I.3 - Correlation Matrix
 cor_matrix <- cor(df[, sapply(df, is.numeric)], use = "complete.obs")
 cor_matrix
 # Melt the correlation matrix for ggplot
 melted_cor_matrix <- melt(cor_matrix, na.rm = TRUE)
 # Plot using ggplot2
 ggplot(melted_cor_matrix, aes(x=Var1, y=Var2, fill=value)) +
   geom_tile() +
   scale_fill_gradient2(low = "red", high = "orange", mid = "white", midpoint = 0) +
   theme_minimal() +
   theme(axis.text.x = element_text(angle = 45, hjust = 1)) +
   labs(title = "Correlation Matrix", x = "", y = "")



#I.5a - Multivariate Regression Modeling
M4 <- lm (Premium.Amount ~ Previous.Claims.History + Income.Level+Driving.Record,Training) #model using "Previous.Claims.History" as the predictor variable and "Premium.Amount" as the output. 
summary(M4)


#BENCHMARKING UNREGULARIZED MODEL PERFORMANCE#
##############################################
#I.5a - In and Out of Sample Error Metric - RMSE#
#################################################

#In-Sample Predictions on Training Data
M4_IN_PRED <- predict(M4,Training)
#Out-Of-Sample Predictions on Validation Data
M4_OUT_PRED <- predict(M4,Validation)

#RMSE for In and Out-Of-Sample
M4_IN_RMSE <- sqrt(sum(M4_IN_PRED-Training$Premium.Amount)^2/length(M4_IN_PRED))
M4_OUT_RMSE <- sqrt(sum(M4_OUT_PRED-Validation$Premium.Amount)^2)/length(M4_OUT_PRED)

##IN AND OUT OF SAMPLE ERROR#

print(M4_IN_RMSE)

print(M4_OUT_RMSE)

#I. 5b - Regularization of Model#
#################################

reg_M4 <-lm.ridge(Premium.Amount ~ .,Training, lambda=seq(0,.5,.01)) #BUILD REGULARIZED MODEL

##DIAGNOSTIC OUTPUT##
summary_reg <- tidy(reg_M4)
summary(summary_reg)
print(summary_reg)

#BENCHMARKING REGULARIZED MODEL PERFORMANCE#
############################################

#I.5b - In and Out of Sample Error Metric - RMSE
#In-Sample Predictions on Training Data
reg_M4_IN_PRED <- predict(M4,Training)
#Out-Of-Sample Predictions on Validation Data
reg_M4_OUT_PRED <- predict(M4,Validation)

#RMSE for In and Out-Of-Sample
reg_M4_IN_RMSE <- sqrt(sum(reg_M4_IN_PRED-Training$Premium.Amount)^2/length(reg_M4_IN_PRED))
reg_M4_OUT_RMSE <- sqrt(sum(reg_M4_OUT_PRED-Validation$Premium.Amount)^2)/length(reg_M4_OUT_PRED)


##IN AND OUT OF SAMPLE ERROR#
print(reg_M4_IN_RMSE)

print(reg_M4_OUT_RMSE)

###TUNING###

kern_type<-"radial" #SPECIFY KERNEL TYPE

tune_control<-tune.control(cross=10) #SET K-FOLD CV PARAMETERS
set.seed(12)
TUNE <- tune.svm(x = training[,-1],
                 y = training[,1],
                 type = "eps-regression",
                 kernel = kern_type,
                 tunecontrol=tune_control,
                 cost=c(.01, .1, 1, 10, 100, 1000), #REGULARIZATION PARAMETER
                 gamma = 1/(ncol(training)-1), #KERNEL PARAMETER
                 coef0 = 0,           #KERNEL PARAMETER
                 degree = 2)          #POLYNOMIAL KERNEL PARAMETER

print(TUNE) #OPTIMAL TUNING PARAMETERS FROM VALIDATION PROCEDURE

#I. 5c - BUILDING NON-LINEAR MODEL USING (POLYNOMIAL) FEATURE TRANSFORMATIONS#
##############################################################################

df$Income.Level2<-df$Income.Level^2 #QUADRATIC TRANSFORMATION (2nd ORDER)
df$Income.Level3<-df$Income.Level^3 #CUBIC TRANSFORMATION (3rd ORDER)
df$Income.Level4<-df$Income.Level^4 #FOURTH ORDER TERM  

#MODEL FORMULA#
model_fmla<-Premium.Amount~Income.Level+Income.Level2+Income.Level3+Income.Level4+Driving.History+Claim.History



#BENCHMARKING NON-LINEAR MODEL PERFORMANCE#
############################################

#I.5c - In and Out of Sample Error Metric - RMSE#
#################################################

#REPORT RMSE IN-SAMPLE AND OUT-OF-SAMPLE
(RMSE_IN_RAND<-mean(e_in))
(RMSE_OUT_RAND<-mean(e_out))

#UNKOWN ERROR NEED ASSISTANCE RESOLVING IN ORDER TO CALCULATE IN AND OUT OF SAMPLE ERROR#

#I.5d ESTIMATING A SUPPORT VECTOR#
##################################

df$Premium.Amount<-as.factor(df$Premium.Amount) #FOR tune.svm()


#VERIFY STRATIFIED SAMPLING YIELDS EQUALLY SKEWED PARTITIONS
mean(training$Premium.Amount==1)
mean(test$Premium.Amount==1)

kern_type<-"radial" #SPECIFY KERNEL TYPE

#BUILD SVM CLASSIFIER
SVM_Model<- svm(Premium.Amount ~ ., 
                data = training, 
                type = "eps-regression", #set to "eps-regression" for numeric prediction
                kernel = kern_type,
                cost=1,                   #REGULARIZATION PARAMETER
                gamma = 1/(ncol(training)-1), #DEFAULT KERNEL PARAMETER
                coef0 = 0,                    #DEFAULT KERNEL PARAMETER
                degree=2,                     #POLYNOMIAL KERNEL PARAMETER
                scale = FALSE)                #RESCALE DATA? (SET TO TRUE TO NORMALIZE)

print(SVM_Model) #DIAGNOSTIC SUMMARY

#REPORT IN AND OUT-OF-SAMPLE ERRORS (1-ACCURACY)
(E_IN_PRETUNE<-1-mean(predict(SVM_Model, training)==training$Income.Level))
(E_OUT_PRETUNE<-1-mean(predict(SVM_Model, test)==test$Income.Level))



#TUNING THE SVM BY CROSS-VALIDATION#
####################################

tune_control<-tune.control(cross=10) #SET K-FOLD CV PARAMETERS
set.seed(12)
TUNE <- tune.svm(x = training[,-1],
                 y = training[,1],
                 type = "eps-regression",
                 kernel = kern_type,
                 tunecontrol=tune_control,
                 cost=c(.01, .1, 1, 10, 100, 1000), #REGULARIZATION PARAMETER
                 gamma = 1/(ncol(training)-1), #KERNEL PARAMETER
                 coef0 = 0,           #KERNEL PARAMETER
                 degree = 2)          #POLYNOMIAL KERNEL PARAMETER

print(TUNE) #OPTIMAL TUNING PARAMETERS FROM VALIDATION PROCEDURE

#RE-BUILD MODEL USING OPTIMAL TUNING PARAMETERS
SVM_Retune<- svm(Premium.Amount ~ ., 
                 data = training, 
                 type = "eps-regression", 
                 kernel = kern_type,
                 degree = TUNE$best.parameters$degree,
                 gamma = TUNE$best.parameters$gamma,
                 coef0 = TUNE$best.parameters$coef0,
                 cost = TUNE$best.parameters$cost,
                 scale = FALSE)

print(SVM_Retune) #DIAGNOSTIC SUMMARY

#REPORT IN AND OUT-OF-SAMPLE ERRORS (1-ACCURACY) ON RETUNED MODEL
(E_IN_RETUNE<-1-mean(predict(SVM_Retune, training)==training$Premium.Amount))
(E_OUT_RETUNE<-1-mean(predict(SVM_Retune, test)==test$Premium_Amount))

#SUMMARIZE RESULTS IN A TABLE:
TUNE_TABLE <- matrix(c(E_IN_PRETUNE, 
                       E_IN_RETUNE,
                       E_OUT_PRETUNE,
                       E_OUT_RETUNE),
                     ncol=2, 
                     byrow=TRUE)

colnames(TUNE_TABLE) <- c('UNTUNED', 'TUNED')
rownames(TUNE_TABLE) <- c('E_IN', 'E_OUT')
TUNE_TABLE #REPORT OUT-OF-SAMPLE ERRORS FOR BOTH HYPOTHESIS


#I.5e ESTIMATING A REGERESSION TREE#
####################################

#SPECIFYING THE CLASSIFICATION TREE MODEL
class_spec <- decision_tree(min_n = 20 , #minimum number of observations for split
                            tree_depth = 30, #max tree depth
                            cost_complexity = 0.01)  %>% #regularization parameter
  set_engine("rpart") %>%
  set_mode("classification")
print(class_spec)

#ESTIMATING THE MODEL (CAN BE DONE IN ONE STEP ABOVE WITH EXTRA %>%)
class_fmla <- Premium.Amount ~ .
class_tree <- class_spec %>%
  fit(formula = class_fmla, data = train)
print(class_tree)

#VISUALIZING THE CLASSIFICATION TREE MODEL:
class_tree$fit %>%
  rpart.plot(type = 4, extra = 2, roundint = FALSE)

plotcp(class_tree$fit)

#GENERATE OUT-OF-SAMPLE PREDICTIONS ON THE TEST SET AND COMBINE WITH TEST DATA
pred_class <- predict(class_tree, new_data = test, type="class") %>%
  bind_cols(test) #ADD CLASS PREDICTIONS DIRECTLY TO TEST DATA

pred_prob <- predict(class_tree, new_data = test, type="prob") %>%
  bind_cols(test) #ADD PROBABILITY PREDICTIONS DIRECTLY TO TEST DATA

#I.5f Estimate a tree-based ensemble model#
###########################################
#MODEL DESCRIPTION:
fmla <- Premium.Amount ~.

##############################
#SPECIFYING BAGGED TREE MODEL#
##############################

spec_bagged <- bag_tree(min_n = 20 , #minimum number of observations for split
                        tree_depth = 30, #max tree depth
                        cost_complexity = 0.01, #regularization parameter
                        class_cost = NULL)  %>% #for output class imbalance adjustment (binary data only)
  set_mode("classification") %>% #can set to regression for numeric prediction
  set_engine("rpart", times=100) #times = # OF ENSEMBLE MEMBERS IN FOREST
spec_bagged

#FITTING THE MODEL
set.seed(123)
bagged_forest <- spec_bagged %>%
  fit(formula = fmla, data = train)
print(bagged_forest)

#GENERATE IN-SAMPLE PREDICTIONS ON THE TRAIN SET AND COMBINE WITH TRAIN DATA
pred_class_bf_in <- predict(bagged_forest, new_data = train, type="class") %>%
  bind_cols(train) #ADD CLASS PREDICTIONS DIRECTLY TO TEST DATA

#GENERATE IN-SAMPLE CONFUSION MATRIX AND DIAGNOSTICS
confusion <- table(pred_class_bf_in$.pred_class, pred_class_bf_in$Premium.Amount)
confusionMatrix(confusion) #FROM CARET PACKAGE

#GENERATE OUT-OF-SAMPLE PREDICTIONS ON THE TEST SET AND COMBINE WITH TEST DATA
pred_class_bf_out <- predict(bagged_forest, new_data = test, type="class") %>%
  bind_cols(test) #ADD CLASS PREDICTIONS DIRECTLY TO TEST DATA

#GENERATE OUT-OF-SAMPLE CONFUSION MATRIX AND DIAGNOSTICS
confusion <- table(pred_class_bf_out$.pred_class, pred_class_bf_out$Premium.Amount)
confusionMatrix(confusion) #FROM CARET PACKAGE

################################
#SPECIFYING RANDOM FOREST MODEL#
################################

spec_rf <- rand_forest(min_n = 20 , #minimum number of observations for split
                       trees = 100, #of ensemble members (trees in forest)
                       mtry = 2)  %>% #number of variables to consider at each split
  set_mode("classification") %>% #can set to regression for numeric prediction
  set_engine("ranger") #alternative engine / package: randomForest
spec_rf

#FITTING THE RF MODEL
set.seed(123) #NEED TO SET SEED WHEN FITTING OR BOOTSTRAPPED SAMPLES WILL CHANGE
random_forest <- spec_rf %>%
  fit(formula = fmla, data = train) #%>%
print(random_forest)

#RANKING VARIABLE IMPORTANCE (CAN BE DONE WITH OTHER MODELS AS WELL)
set.seed(123) #NEED TO SET SEED WHEN FITTING OR BOOTSTRAPPED SAMPLES WILL CHANGE
rand_forest(min_n = 20 , #minimum number of observations for split
            trees = 100, #of ensemble members (trees in forest)
            mtry = 2)  %>% #number of variables to consider at each split
  set_mode("classification") %>%
  set_engine("ranger", importance = "impurity") %>%
  fit(fmla, data = train) %>%
  vip() #FROM VIP PACKAGE - ONLY WORKS ON RANGER FIT DIRECTLY

#GENERATE IN-SAMPLE PREDICTIONS ON THE TRAIN SET AND COMBINE WITH TRAIN DATA
pred_class_rf_in <- predict(random_forest, new_data = train, type="class") %>%
  bind_cols(train) #ADD CLASS PREDICTIONS DIRECTLY TO TEST DATA

#GENERATE IN-SAMPLE CONFUSION MATRIX AND DIAGNOSTICS
confusion <- table(pred_class_rf_in$.pred_class, pred_class_rf_in$Premium.Amount)
confusionMatrix(confusion) #FROM CARET PACKAGE

#GENERATE OUT-OF-SAMPLE PREDICTIONS ON THE TEST SET AND COMBINE WITH TEST DATA
pred_class_rf_out <- predict(random_forest, new_data = test, type="class") %>%
  bind_cols(test) #ADD CLASS PREDICTIONS DIRECTLY TO TEST DATA

#GENERATE OUT-OF-SAMPLE CONFUSION MATRIX AND DIAGNOSTICS
confusion <- table(pred_class_rf_out$.pred_class, pred_class_rf_out$Premium.Amount)
confusionMatrix(confusion) #FROM CARET PACKAGE


#I.5g Create a table summarizing the in-sample and out-of-sample estimated performance for each of the models#
##############################################################################################################





