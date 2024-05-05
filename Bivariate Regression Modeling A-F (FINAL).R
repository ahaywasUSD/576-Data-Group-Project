#LOAD LIBRARIES
library(dplyr)
library(ggplot2)
library(reshape2)
library(caret)
library(lmridge) #for Ridge Regression
library(broom)
library(MASS)
library(mgcv) #for GAM
library(e1071)
library(glmnet) #For regularization

#IMPORT THE DATA
df <- read.csv('https://raw.githubusercontent.com/ahaywasUSD/576-Data-Group-Project/main/data_synthetic_ATH.csv')
#CONVERT TO FACTORS
df <- df %>%
  mutate_if(is.character, as.factor)
summary(df)
#Nonlinear Transformations of Previous.Claims.History for I4.B
df$Previous.Claims.History2 <- df$Previous.Claims.History^2 #polynomial transformation
df$Previous.Claims.History3 <- df$Previous.Claims.History^3 #cubic transformation
df$Previous.Claims.History4 <- df$Previous.Claims.History^4 #4th order transformation
df$Previous.Claims.History_LN <- log(df$Previous.Claims.History) #logarithmic transformation
summary(df)

##########################################################
##############I.1 - Data Partitioning#####################
##########################################################
 set.seed(123)
 #FRACTION OF DATA TO BE USED AS IN-SAMPLE TRAINING DATA
 p<-.7 #70% FOR TRAINING (IN-SAMPLE)
 #h<-.3 #HOLDOUT (FOR TESTING OUT-OF-SAMPLE)
 
 obs_count <- dim(df)[1] #TOTAL OBSERVATIONS IN DATA
 
 #OF OBSERVATIONS IN THE TRAINING DATA (IN-SAMPLE DATA)
 training_size <- floor(p * obs_count)
 
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
 
 ##########################################################
 #############I.3 - Correlation Matrix#####################
 ##########################################################
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

 ##########################################################
 ##########I.4 - Bivariate Regression Modeling#############
 ##########################################################
 
 ##########I.4a - Linear Model Benchmarking################
 ##########################################################
 #Linear Model
M1 <- lm(Premium.Amount ~ Previous.Claims.History,Training) #model using "Previous.Claims.History" as the predictor variable and "Premium.Amount" as the output. 
summary(M1)
 
 #In-Sample Predictions on Training Data
 M1_IN_PRED <- predict(M1,Training)
 #Out-Of-Sample Predictions on Validation Data
 M1_OUT_PRED <- predict(M1,Validation)
 
##Computing In and Out of Sample Error Metric - RMSE
M1_IN_RMSE <- sqrt(sum(M1_IN_PRED-Training$Premium.Amount)^2/length(M1_IN_PRED))
M1_OUT_RMSE <- sqrt(sum(M1_OUT_PRED-Validation$Premium.Amount)^2)/length(M1_OUT_PRED)

print(M1_IN_RMSE)
print(M1_OUT_RMSE)

##########################################################
###I.4b - Bivariate Model on Nonlinear Transformations####
##########################################################
#Nonlinear Transformations of Previous.Claims.History
df$Previous.Claims.History2 <- df$Previous.Claims.History^2 #polynomial transformation
df$Previous.Claims.History3 <- df$Previous.Claims.History^3 #cubic transformation
df$Previous.Claims.History4 <- df$Previous.Claims.History^4 #4th order transformation
df$Previous.Claims.History_LN <- log(df$Previous.Claims.History) #logarithmic transformation
summary(df)

#Bivariate model
M2 <- lm(Premium.Amount ~ Previous.Claims.History + Previous.Claims.History2, Training)
summary(M2)

#Nonlinear Predictions on Training (In-Sample)
M2_IN_PRED <- predict(M2, Training)
#Nonlinear Predictions on Validation Data (Out-Of-Sample)
M2_OUT_PRED <- predict(M2, Validation)

#RMSE For M2 In and Out-Of Sample Data
M2_IN_RMSE <- sqrt(sum(M2_IN_PRED-Training$Premium.Amount)^2)/length(M2_IN_PRED)
M2_OUT_RMSE <- sqrt(sum(M2_OUT_PRED-Validation$Premium.Amount)^2)/length(M2_OUT_PRED)

#M3
M3 <- lm(Premium.Amount ~ Previous.Claims.History + Previous.Claims.History2 + Previous.Claims.History3, Training)
summary(M3)

#Nonlinear Predictions on Training (In-Sample)
M3_IN_PRED <- predict(M3, Training)
#Nonlinear Predictions on Validation Data (Out-Of-Sample)
M3_OUT_PRED <- predict(M3, Validation)

#RMSE For M3 In and Out-Of Sample Data
M3_IN_RMSE <- sqrt(sum(M3_IN_PRED-Training$Premium.Amount)^2)/length(M3_IN_PRED)
M3_OUT_RMSE <- sqrt(sum(M3_OUT_PRED-Validation$Premium.Amount)^2)/length(M3_OUT_PRED)


###########MODEL COMPARISON###########
#Table to easily compare RMSE across different models
TABLE_VAL_1 <- as.table(matrix(c(M1_IN_RMSE,M2_IN_RMSE,M3_IN_RMSE,M1_OUT_RMSE,M2_OUT_RMSE,M3_OUT_RMSE), ncol=3, byrow=TRUE))
colnames(TABLE_VAL_1) <- c('LINEAR', 'QUADRATIC','CUBIC') #Column Headers
rownames(TABLE_VAL_1) <- c('RMSE_IN', 'RMSE_OUT') #Row Labels
TABLE_VAL_1 #Reporting In and Out-Of-Sample ERRORS for different models.

##########################################################
####I4.C - Regularization using Ridge Regression##########
##########################################################
#########
x <- as.matrix(Training[,c("Previous.Claims.History","Previous.Claims.History2")])
y <- Training$Premium.Amount

#Fit model with Lasso
set.seed(123)
lasso_model <- glmnet (x,y,alpha = 1)
# Optionally, you can use cross-validation to find the best lambda
cv_lasso <- cv.glmnet(x, y, alpha = 1)
cv_lasso$rmse <- sqrt(cv_lasso$cvm)
plot(cv_lasso$lambda, cv_lasso$rmse, xlab = "Lambda", ylab = "RMSE",main="RMSE vs. Lambda in Lasso Regression")
best_lambda_lasso <- cv_lasso$lambda[which.min(cv_lasso$rmse)]
print(best_lambda_lasso)
######
#######
# Fit the Ridge model
ridge_model <- glmnet(x, y, alpha = 0)  # alpha = 0 for Ridge

# Use cross-validation to find the best lambda
cv_ridge <- cv.glmnet(x, y, type.measure = "mse", alpha = 0)
cv_ridge$rmse <- sqrt(cv_ridge$cvm)
plot(cv_ridge$lambda, cv_ridge$rmse, xlab = "Lambda", ylab = "RMSE", main = "RMSE vs. Lambda in Ridge Regression")
best_lambda_ridge <- cv_ridge$lambda[which.min(cv_ridge$rmse)]
print(best_lambda_ridge)
#######
# Check coefficients at the best lambda
coef(lasso_model, s = best_lambda_lasso)
coef(ridge_model, s = best_lambda_ridge)

#In-Sample Predictions
x_training <- as.matrix(Training[,c("Previous.Claims.History","Previous.Claims.History2")])
predictions_train <- predict(ridge_model, newx=x_training)
#In-Sample Ridge RMSE
RMSE_IN_ridge <- sqrt(sum(predictions_train - Training$Premium.Amount)^2)/length(predictions_train)
print(RMSE_IN_ridge)


#Out-Of-Sample Ridge Predictions
x_validation <- as.matrix(Validation[,c("Previous.Claims.History","Previous.Claims.History2")])
predictions_validation <- predict(ridge_model, newx=x_validation)
#Out-Of-Sample Ridge RMSE
RMSE_OUT_ridge <- sqrt(sum(predictions_validation - Validation$Premium.Amount)^2)/length(predictions_validation)
print(RMSE_OUT_ridge)

#Table to easily compare RMSE across different ridge models
TABLE_VAL_ridge <- as.table(matrix(c(M2_IN_RMSE,RMSE_IN_ridge,M2_OUT_RMSE,RMSE_OUT_ridge), ncol=2, byrow=TRUE))
colnames(TABLE_VAL_ridge) <- c('Quadratic','RIDGE') #Column Headers
rownames(TABLE_VAL_ridge) <- c('RMSE_IN', 'RMSE_OUT') #Row Labels
TABLE_VAL_ridge #Reporting In and Out-Of-Sample ERRORS for different models.

##########################################################
###TUNING RIDGE
set.seed(123)
best_lambda_ridge <- cv_ridge$lambda[which.min(cv_ridge$rmse)]
print(best_lambda_ridge)
optimal_ridge_model <- glmnet (x, y, alpha = 0, lambda = best_lambda_ridge)

print(coef(optimal_ridge_model))

#Validate
x_validation <- as.matrix(Validation[,c("Previous.Claims.History", "Previous.Claims.History2")])
y_validation <- Validation$Premium.Amount

#Optimal In-Sample Predictions & RMSE
Optimal_predictions <- predict(optimal_ridge_model, s = best_lambda_ridge, newx=x_training)
RMSE_IN_Optimalridge <- sqrt(sum(Optimal_predictions-Training$Premium.Amount)^2)/length(Optimal_predictions)
print(RMSE_IN_Optimalridge)

#Optimal Out-Of-Sample Predictions & RMSE
Optimal_predictions_val <- predict(optimal_ridge_model, s=best_lambda_ridge,newx=x_validation, newy=y_validation)
RMSE_OUT_Optimalridge <- sqrt(sum(Optimal_predictions_val-Validation$Premium.Amount)^2)/length(Optimal_predictions_val)
print(RMSE_OUT_Optimalridge)

#Table to easily compare RMSE across different ridge models
TABLE_VAL_Tuned_Ridge <- as.table(matrix(c(M2_IN_RMSE,RMSE_IN_ridge,RMSE_IN_Optimalridge,M2_OUT_RMSE,RMSE_OUT_ridge,RMSE_OUT_Optimalridge), ncol=3, byrow=TRUE))
colnames(TABLE_VAL_Tuned_Ridge) <- c('Quadratic','RIDGE','TUNED RIDGE') #Column Headers
rownames(TABLE_VAL_Tuned_Ridge) <- c('RMSE_IN', 'RMSE_OUT') #Row Labels
TABLE_VAL_Tuned_Ridge #Reporting In and Out-Of-Sample ERRORS for different models.
##########################################################


##########################################################
###I4.D - Incorporating Generalized Additive Structure####
##########################################################

#MODEL 5: Generalized Additive Model
mean(Training$Premium.Amount) #CHECK THE MEAN
var(Training$Premium.Amount) #CHECK THE VARIANCE
#NOTE THEY ARE NOT EQUAL, QUASIPOISSON REGRESSION DOES NOT REQUIRE THE MEAN AND VARIANCE OF THE OUTPUT
#VARIABLE BE EQUAL, SO WE WILL USE QUASIPOISSON.
#GAM ON LINEAR MODEL
M4 <- gam(Premium.Amount ~ s(Previous.Claims.History, k=4) + s(Previous.Claims.History2, k =4), data = Training)
summary(M4) #generates summary diagnostic output

#plot
par(mfrow=c(2,2))
plot(M4)

#GENERATING PREDICTIONS ON THE TRAINING DATA
M4_IN_PRED <- predict(M4, Training) #generate predictions on the (in-sample) training data

#GENERATING PREDICTIONS ON THE TEST DATA FOR BENCHMARKING
M4_OUT_PRED <- predict(M4, Validation) #generate predictions on the (out-of-sample) testing data

#COMPUTING / REPORTING IN-SAMPLE AND OUT-OF-SAMPLE ROOT MEAN SQUARED ERROR
(M4_IN_RMSE<-sqrt(sum((M4_IN_PRED-Training$Premium.Amount)^2)/length(M4_IN_PRED)))  #computes in-sample error
(M4_OUT_RMSE<-sqrt(sum((M4_OUT_PRED-Validation$Premium.Amount)^2)/length(M4_OUT_PRED))) #computes out-of-sample 

#Table to easily compare RMSE across different ridge models
TABLE_VAL_GAM <- as.table(matrix(c(M2_IN_RMSE,RMSE_IN_ridge,RMSE_IN_Optimalridge,M4_IN_RMSE,M2_OUT_RMSE,RMSE_OUT_ridge,RMSE_OUT_Optimalridge,M4_OUT_RMSE), ncol=4, byrow=TRUE))
colnames(TABLE_VAL_GAM) <- c('Quadratic','RIDGE','TUNED RIDGE','GAM') #Column Headers
rownames(TABLE_VAL_GAM) <- c('RMSE_IN', 'RMSE_OUT') #Row Labels
TABLE_VAL_GAM #Reporting In and Out-Of-Sample ERRORS for different models.



##########################################################
########I4.E - Plotting#######
##########################################################


##########################################################
####I4.F - Table Summary###
##########################################################
# Creating a summary table of RMSE values
#Training RMSE
M1_IN_RMSE
M2_IN_RMSE
RMSE_IN_ridge
RMSE_IN_Optimalridge
M4_IN_RMSE
#Validation RMSE
M1_OUT_RMSE
M2_OUT_RMSE
RMSE_OUT_ridge
RMSE_OUT_Optimalridge
M4_OUT_RMSE

validation_rmse <- c(M1 = M1_OUT_RMSE, M2 = M2_OUT_RMSE, RMSE_OUT_ridge,RMSE_OUT_Optimalridge,M4 = M4_OUT_RMSE)
best_model_name <- names(which.min(validation_rmse))
best_model_rmse <- min(validation_rmse)

#Uncontaminated Out-Of-Sample RMSE on best model
# Assuming predictions function is aligned with the best model selected
uncontaminated_predictions <- predict(get(best_model_name), newdata = Testing)
rmse_uncontaminated <- sqrt(sum((uncontaminated_predictions - Testing$Premium.Amount)^2)/length(uncontaminated_predictions))

# Create summary table
summary_table <- data.frame(
  Model = c("M1", "M2", "Ridge", "Optimal_Ridge","GAM"),
  RMSE_Training = c(M1_IN_RMSE, M2_IN_RMSE, RMSE_IN_ridge,RMSE_IN_Optimalridge, M4_IN_RMSE),
  RMSE_Validation = c(M1_OUT_RMSE,M2_OUT_RMSE,RMSE_OUT_ridge,RMSE_OUT_Optimalridge,M4_OUT_RMSE),
  stringsAsFactors = FALSE
)

# Add uncontaminated RMSE for the best model
summary_table$RMSE_Uncontaminated <- ifelse(summary_table$Model == best_model_name, rmse_uncontaminated, NA)

# Print table
print(summary_table)

# Print best model based on Validation data
cat("Best Model Based on Validation Data: ", best_model_name, "with RMSE: ", best_model_rmse, "\n")
cat("Uncontaminated Test RMSE for", best_model_name, ":", rmse_uncontaminated)

