#PROJECT PART 5#
################

#LOAD LIBRARIES
library(ggplot2)
library(reshape2)
library(caret)
library(lmridge) #for Ridge Regression
library(broom)
library(MASS)
library(broom) #FOR glance() AND tidy()
library(Metrics) #FOR rmse()
library(e1071) #SVM LIBRARY
library(rsample) #FOR initial_split() STRATIFIED RANDOM SAMPLING
library(tidymodels) #INCLUDES parsnip PACKAGE FOR decision_tree()
library(caret) #FOR confusionMatrix()
library(rpart.plot)
library(tidymodels)
library(baguette) #FOR BAGGED TREES
library(caret) #FOR confusionMatrix()
library(vip) #FOR VARIABLE IMPORTANCE
library(kernlab)
library(ggplot2)
library(readxl)
library(dplyr)
library(glmnet)
library(rpart)
library(randomForest)

#IMPORT THE DATA
df <- read.csv('https://raw.githubusercontent.com/ahaywasUSD/576-Data-Group-Project/main/data_synthetic_ATH.csv')
#CONVERT TO FACTORS
df <- df %>%
  mutate_if(is.character, as.factor)
summary(df)

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

## Ridge Model ##
#################

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

#Visualization of the data#
ggplot(df, aes(x = Driving.Record, y = Premium.Amount)) +
  geom_bar(stat = "identity", fill = "skyblue") +
  labs(title = "Driving Record v Premium Amount", x = "Driving Record", y = "Premium Amount") +
  theme_minimal()

ggplot(df, aes(x = Income.Level, y = Premium.Amount)) +
  geom_bar(stat = "identity", fill = "orange") +
  labs(title = "Income Level v Premium Amount", x = "Income Level", y = "Premium Amount") +
  theme_minimal()

ggplot(df, aes(x = Previous.Claims.History, y = Premium.Amount)) +
  geom_bar(stat = "identity", fill = "green") +
  labs(title = "Previous.Claims.History v Premium Amount", x = "Previous Claims History", y = "Premium Amount") +
  theme_minimal()

## Plotting Model Complexity (Lambda) vs Validation Error ##
ggplot(summary_reg, aes(lambda, GCV)) +
  geom_line(color = "blue", size = 1) +  # Blue line for the plot
  geom_vline(xintercept = summary_reg $lambdaGCV, col = 'red', lty = 2) +  # Red dashed line for the optimal lambda
  labs(x = "Lambda (Regularization Strength)", y = "Generalized Cross Validation (GCV)", 
       title = "Model Complexity vs Validation Error") +  # Labels for axes and title
  theme_minimal() +  # Minimal theme for the plot
  theme(plot.title = element_text(face = "bold", size = 14),  # Title style
        axis.title = element_text(face = "bold", size = 12),  # Axis title style
        axis.text = element_text(size = 10))  # Axis text style


###TUNING###

## Build regularized model using ridge regression with cross-validation ##
Tuned_summary_reg <- cv.glmnet(
  x = as.matrix(Training[, -1]),  # Features matrix (excluding response variable)
  y = Training$Premium.Amount,  # Response variable
  alpha = 0,  # Ridge regression
  lambda = seq(0, 0.5, 0.01),  # Range of lambda values
  nfolds = 5,  # 5-fold cross-validation
  type.measure = "mse"  # Measure to optimize (mean squared error)
)

## Extract optimal lambda ##
optimal_lambda <- Tuned_summary_reg$lambda.min

## Build final regularized model using optimal lambda ##
final_reg <- glmnet(
  x = as.matrix(Training[, -1]),  # Features matrix (excluding response variable)
  y = Training$Premium.Amount,  # Response variable
  alpha = 0,  # Ridge regression
  lambda = optimal_lambda  # Optimal lambda from cross-validation
)

## Extract model coefficients for the final regularized model ##
final_reg$beta

## Summary of the final regularized model ##
summary(final_reg)

## Extract model metrics using broom's glance function ##
glance(final_reg)

## Convert summary to a data frame ##
summary_df <- as.data.frame(tidy(final_reg))
head(summary_df, 10)


Tuned_summary_reg <- glmnet(
  x = as.matrix(Training[, -1]),  # Features matrix (excluding response variable)
  y = Training$Premium.Amount,  # Response variable
  alpha = 0,  # Ridge regression
  lambda = optimal_lambda  # Optimal lambda from cross-validation
)

## Make predictions on both training and testing sets
pred_train <- predict(Tuned_summary_reg, s = optimal_lambda, newx = as.matrix(Training[, -1]), type = "response")
pred_test <- predict(Tuned_summary_reg, s = optimal_lambda, newx = as.matrix(Validation[, -1]), type = "response")

## Calculate RMSE for in-sample and out-of-sample predictions
ridge_rmse_train <- sqrt(mean((pred_train - Training$Premium.Amount)^2))
ridge_rmse_test <- sqrt(mean((pred_test - Validation$Premium.Amount)^2))

## Print RMSE values

print(ridge_rmse_train)

print(ridge_rmse_test)


#I. 5c - BUILDING NON-LINEAR MODEL USING (POLYNOMIAL) FEATURE TRANSFORMATIONS#
##############################################################################

df$Income.Level2<-df$Income.Level^2 #QUADRATIC TRANSFORMATION (2nd ORDER)
df$Income.Level3<-df$Income.Level^3 #CUBIC TRANSFORMATION (3rd ORDER)
df$Income.Level4<-df$Income.Level^4 #FOURTH ORDER TERM  

# Fit the model
Poly_model <- lm(Premium.Amount ~ Income.Level + Income.Level^2 + Income.Level^3 + Income.Level^4 + Training$Driving.Record + Training$Claim.History, data = Training)


#In and Out of Sample Error Metric - RMSE#
##########################################

# In-sample predictions
in_sample_predictions <- predict(Poly_model)

# Out-of-sample predictions
out_of_sample_predictions <- predict(Poly_model, Training)

#BENCHMARKING NON-LINEAR MODEL PERFORMANCE#
############################################

#Comparing performance#
#######################

print(ridge_rmse_train)
print(ridge_rmse_test)
print(in_sample_predictions)
print(out_of_sample_predictions)


##I. 5d - ESTIMATING A SUPPORT VECTOR#
######################################

## Convert to factor for SVM
Training$Driving.Record <- as.factor(Training$Driving.Record)
Validation$Driving.Record <- as.factor(Validation$Driving.Record)

## VERIFY STRATIFIED SAMPLING YIELDS EQUALLY SKEWED PARTITIONS
mean(Training$Driving.Record == 1)
mean(Validation$Driving.Record == 1)

kern_type <- "radial"  # Specify kernel type

## BUILD SVM CLASSIFIER
SVM_Model <- svm(Driving.Record ~ .,
                 data = Training,
                 type = "C-classification",  # Set to "C-classification" for classification
                 kernel = kern_type,
                 cost = 1,  # Regularization parameter
                 gamma = 1/(ncol(Training)-1),  # Default kernel parameter
                 coef0 = 0,  # Default kernel parameter
                 degree = 2,  # Polynomial kernel parameter
                 scale = FALSE)  

print(SVM_Model)  # Diagnostic summary


## REPORT IN AND OUT-OF-SAMPLE ERRORS (1-ACCURACY)
(E_IN_PRETUNE <- 1 - mean(predict(SVM_Model, Training) == Training$Driving.Record))
(E_OUT_PRETUNE <- 1 - mean(predict(SVM_Model, Validation) == Validation$Driving.Record))


################################################################################################

#TUNING THE SVM BY CROSS-VALIDATION#
####################################

#RE-BUILD MODEL USING OPTIMAL TUNING PARAMETERS
Training$Driving.Record <- as.factor(Training$Driving.Record)
Validation$Driving.Record <- as.factor(Validation$Driving.Record)

## Specify the parameter grid for tuning
tune_grid <- expand.grid(sigma = c(0.01, 0.1, 1),
                         C = c(0.1, 1, 10))

## Define cross-validation control
ctrl <- trainControl(method = "cv",  # Cross-validation method
                     number = 5,  # Number of folds
                     verboseIter = TRUE)  # Print progress during tuning

## Tune the SVM model using caret's tune function
tuned_model <- train(Driving.Record ~ .,
                     data = Training,
                     method = "svmRadial",  # SVM with radial kernel
                     trControl = ctrl,  # Cross-validation control
                     tuneGrid = tune_grid,  # Hyperparameter grid
                     preProcess = c("center", "scale"),  # Preprocessing
                     metric = "Accuracy")  # Metric to optimize

## Print the tuned model
print(tuned_model)

## Extract the best model from the tuning results
best_model <- tuned_model$finalModel

## Make predictions on training and validation sets
train_predictions <- predict(tuned_model, Training)
validation_predictions <- predict(tuned_model, Validation)

## Calculate in-sample and out-of-sample errors
(E_IN_TUNE <- 1 - mean(train_predictions == Training$Driving.Record))
(E_OUT_TUNE <- 1 - mean(validation_predictions == Validation$Driving.Record))


## PRINT SVM MODEL RESULTS ##
cat("\nSVM Model Performance Comparison:\n")
cat("--------------------------------\n")
cat("Model               Pre-Tuned     Tuned\n")
cat("--------------------------------\n")
cat("In-Sample Error     ", sprintf("%.4f", E_IN_PRETUNE), "    ", sprintf("%.4f", E_IN_TUNE), "\n")
cat("Out-Of-Sample Error ", sprintf("%.4f", E_OUT_PRETUNE), "    ", sprintf("%.4f", E_OUT_TUNE), "\n")

########################################################################################################################


#I.5e ESTIMATING A REGERESSION TREE#
####################################

## Fit the regression tree model
regression_tree_model <- rpart(Premium.Amount ~ . - Driving.Record,  # Adjust this formula based on predictors
                               data = Training,
                               minsplit = 20,  # Minimum number of observations for split
                               maxdepth = 30,  # Maximum tree depth
                               cp = 0.0001)  # Complexity parameter for regularization


Testing$Driving.Record <- factor(Testing$Driving.Record)

## Generate out-of-sample predictions on the testing set
pred_regression <- predict(regression_tree_model,Testing)

## Calculate RMSE for in-sample and out-of-sample predictions
regression_rmse_in <- sqrt(mean((pred_regression - Testing$Premium.Amount)^2))
regression_rmse_out <- sqrt(mean((pred_regression - Testing$Premium.Amount)^2))

## Print RMSE values
cat("RMSE for in-sample predictions using regression tree:", regression_rmse_in, "\n")
cat("RMSE for out-of-sample predictions using regression tree:", regression_rmse_out, "\n")

## Visualize the regression tree model
plot(regression_tree_model, uniform = TRUE, main = "Regression Tree Model")
text(regression_tree_model, cex = 0.7, pos = 2)


#######################################################################################################

#Tuning the regression tree#
############################

# Define the control parameters for cross-validation
ctrl <- trainControl(method = "cv",    # Cross-validation method
                     number = 10,      # Number of folds
                     verboseIter = TRUE,  # Print progress
                     search = "grid")  # Search method

# Define the hyperparameter grid for tuning
tune_grid <- expand.grid(
  minsplit = c(10, 20, 30),   # Minimum number of observations for split
  maxdepth = c(20, 30, 40),   # Maximum tree depth
  cp = c(0.001, 0.01, 0.1)    # Complexity parameter for regularization
)

# Train the regression tree model using cross-validation
tuned_regression_tree_model <- train(
  Premium.Amount ~ . - Driving.Record,  # Adjust this formula based on predictors
  data = Training,
  method = "rpart",         # Regression tree method
  trControl = ctrl,         # Cross-validation control parameters
  tuneGrid = tune_grid      # Grid of hyperparameters to search over
)

# Print the tuned model
print(tuned_regression_tree_model)

# Generate out-of-sample predictions on the testing set
pred_regression <- predict(tuned_regression_tree_model, Testing)

# Calculate RMSE for out-of-sample predictions
regression_rmse_out <- sqrt(mean((pred_regression - Testing$Premium.Amount)^2))

# Print RMSE value
cat("RMSE for out-of-sample predictions using tuned regression tree:", regression_rmse_out, "\n")

# Visualize the tuned regression tree model
plot(tuned_regression_tree_model$finalModel, uniform = TRUE, main = "Tuned Regression Tree Model")
text(tuned_regression_tree_model$finalModel, cex = 0.7, pos = 2)


###################################################################################################################

#I.5f Estimate a tree-based ensemble model#
###########################################

################################
#SPECIFYING RANDOM FOREST MODEL#
################################


# Define the control parameters for cross-validation
ctrl <- trainControl(method = "cv",    # Cross-validation method
                     number = 10,      # Number of folds
                     verboseIter = TRUE,  # Print progress
                     search = "grid",  # Search method
                     summaryFunction = defaultSummary)  # Summary function for performance metrics

# Define the hyperparameter grid for tuning
tune_grid <- expand.grid(
  mtry = c(2, 3, 4)  # Number of variables randomly sampled as candidates at each split
)

# Train the Random Forest model using cross-validation
rf_model <- train(
  Premium.Amount ~ . - Driving.Record,  
  data = Training,
  method = "rf",             # Random Forest method
  trControl = ctrl,          # Cross-validation control parameters
  tuneGrid = tune_grid,      # Grid of hyperparameters to search over
  metric = "RMSE"            # Metric to optimize
)

# Print the tuned model
print(rf_model)

# Generate out-of-sample predictions on the testing set
pred_rf <- predict(rf_model, Testing)

# Calculate RMSE for out-of-sample predictions
rf_rmse_out <- sqrt(mean((pred_rf - Testing$Premium.Amount)^2))

# Print RMSE value
cat("RMSE for out-of-sample predictions using Random Forest:", rf_rmse_out, "\n")




##########################################################################################################################

##############################
#SPECIFYING BAGGED TREE MODEL#
##############################

# Define the control parameters for cross-validation
ctrl <- trainControl(method = "cv",    # Cross-validation method
                     number = 10,      # Number of folds
                     verboseIter = TRUE,  # Print progress
                     search = "grid",  # Search method
                     summaryFunction = defaultSummary)  # Summary function for performance metrics

# Define the hyperparameter grid for tuning
tune_grid <- expand.grid(
  mtry = c(2, 3, 4)  # Number of variables randomly sampled as candidates at each split
)

# Train the Bagged Tree model using cross-validation
bagged_model <- train(
  Premium.Amount ~ . - Driving.Record,  
  data = Training,
  method = "treebag",         # Bagged Tree method
  trControl = ctrl,           # Cross-validation control parameters
  tuneGrid = tune_grid,       # Grid of hyperparameters to search over
  metric = "RMSE"             # Metric to optimize
)

# Print the tuned model
print(bagged_model)

# Generate out-of-sample predictions on the testing set
pred_bagged <- predict(bagged_model, Testing)

# Calculate RMSE for out-of-sample predictions
bagged_rmse_out <- sqrt(mean((pred_bagged - Testing$Premium.Amount)^2))

# Print RMSE value
cat("RMSE for out-of-sample predictions using Bagged Tree:", bagged_rmse_out, "\n")




#I.5g Create a table summarizing the in-sample and out-of-sample estimated performance for each of the models#
##############################################################################################################

## OUT OF SAMPLE RESULTS ##
###########################


rmse_values_out <- c( M4_OUT_RMSE,reg_M4_OUT_RMSE,E_OUT_PRETUNE, regression_rmse_out, rf_rmse_out, bagged_rmse_out)  

model_names_out <- c("Unregularized Model OUT","Regularized Model OUT","SVM OUT", "Regression Tree OUT", "Random Forest OUT", "Bagged Forest OUT")  

metric_names_out <- c("RMSE")  

rmse_matrix_out <- matrix(rmse_values, nrow = length(model_names), ncol = length(metric_names),
                      dimnames = list(model_names, metric_names))

# Convert the matrix to a data frame
rmse_df_out <- as.data.frame(rmse_matrix_out)

# Print the data frame
print(rmse_df_out)

## IN SAMPLE RESULTS

rmse_values_in <- c( M4_IN_RMSE,reg_M4_IN_RMSE,E_IN_PRETUNE, regression_rmse_in, rf_rmse_in, bagged_rmse_in)  

model_names_in <- c("Unregularized Model IN","Regularized Model IN","SVM IN", "Regression Tree IN", "Random Forest IN", "Bagged Forest IN") 

metric_names_in <- c("RMSE")  

rmse_matrix_in <- matrix(rmse_values, nrow = length(model_names), ncol = length(metric_names),
                      dimnames = list(model_names, metric_names))

# Convert the matrix to a data frame
rmse_df_in <- as.data.frame(rmse_matrix_in)

# Print the data frame
print(rmse_df_in)


