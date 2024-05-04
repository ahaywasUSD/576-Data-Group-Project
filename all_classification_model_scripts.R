################################################################################
##This is the standard Classification Tree Model using the default parameters.##
################################################################################

# LOAD THE LIBRARIES
library(dplyr)
library(tidymodels) #INCLUDES parsnip PACKAGE FOR decision_tree()
library(caret) #FOR confusionMatrix()
library(rpart.plot)
install.packages("progress")
library(progress)
progress_bar

# IMPORTING THE DATA
df <- read.csv("https://github.com/ahaywasUSD/576-Data-Group-Project/raw/main/data_synthetic_ATH.csv")
#CONVERT TO FACTORS
#df <- df %>%
#mutate_if(is.character, as.factor)
df$Risk.Profile<-as.factor(df$Risk.Profile) #CONVERT OUTPUT TO FACTOR 

##PARTITIONING THE DATA##
# CREATE INDICES FOR DATA PARTITIONING
set.seed(123)
train_index <- createDataPartition(df$Risk.Profile, p = 0.7, list = FALSE, times = 1)
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

# SPECIFYING THE CLASSIFICATION TREE MODEL
class_spec <- decision_tree(min_n = 20, #minimum number of observations for split
                            tree_depth = 30, #max tree depth
                            cost_complexity = 0.01)  %>% #regularization parameter
  set_engine("rpart") %>%
  set_mode("classification")
print(class_spec)

#ESTIMATING THE MODEL (CAN BE DONE IN ONE STEP ABOVE WITH EXTRA %>%)
class_fmla <- Risk.Profile ~ Premium.Amount + Credit.Score + Age
class_tree <- class_spec %>%
  fit(formula = class_fmla, data = train_set)
print(class_tree)

#VISUALIZING THE CLASSIFICATION TREE MODEL:
class_tree$fit %>%
  rpart.plot(type = 1, extra = 2, roundint = FALSE, box.palette = "Blues")

plotcp(class_tree$fit)

#GENERATE IN-SAMPLE PREDICTIONS ON THE TRAIN SET AND COMBINE WITH TRAIN DATA
pred_class_in_CART <- predict(class_tree, new_data = train_set, type="class") %>%
  bind_cols(train_set) #ADD CLASS PREDICTIONS DIRECTLY TO TEST DATA

#GENERATE IN-SAMPLE CONFUSION MATRIX AND DIAGNOSTICS
confusion_in_CART <- table(pred_class_in_CART$.pred_class, pred_class_in_CART$Risk.Profile)
confusionMatrix(confusion_in_CART) #FROM CARET PACKAGE

#GENERATE OUT-OF-SAMPLE PREDICTIONS ON THE TEST SET AND COMBINE WITH TEST DATA
pred_class_out_CART <- predict(class_tree, new_data = test_set, type="class") %>%
  bind_cols(test_set) #ADD CLASS PREDICTIONS DIRECTLY TO TEST DATA

#GENERATE OUT-OF-SAMPLE CONFUSION MATRIX AND DIAGNOSTICS
confusion_out_CART <- table(pred_class_out_CART$.pred_class, pred_class_out_CART$Risk.Profile)
confusionMatrix(confusion_out_CART) #FROM CARET PACKAGE

################################################################################

################################################################################
##This is the standard Classification Tree Model & auto-tuning (best params). ##
################################################################################

#LOADING THE LIBRARIES
library(dplyr)
library(tidymodels) #INCLUDES parsnip PACKAGE FOR decision_tree()
library(caret) #FOR confusionMatrix()
library(rpart.plot)
install.packages("progress")
library(progress)
progress_bar

#IMPORTING THE DATA
df <- read.csv("https://github.com/ahaywasUSD/576-Data-Group-Project/raw/main/data_synthetic_ATH.csv")
#CONVERT TO FACTORS
#df <- df %>%
#mutate_if(is.character, as.factor)
df$Risk.Profile<-as.factor(df$Risk.Profile) #CONVERT OUTPUT TO FACTOR

##PARTITIONING THE DATA##
# CREATE INDICES FOR DATA PARTITIONING
set.seed(123)
train_index <- createDataPartition(df$Risk.Profile, p = 0.7, list = FALSE, times = 1)
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

#MODEL DESCRIPTION
fmla <- Risk.Profile ~ Premium.Amount + Credit.Score + Age

#SPECIFYING AND FITTING THE CLASSIFICATION TREE MODEL WITH DEFAULT PARAMETERS
default_tree <- decision_tree(min_n = 20 , #minimum number of observations for split
                              tree_depth = 30, #max tree depth
                              cost_complexity = 0.01)  %>% #regularization parameter
  set_engine("rpart") %>%
  set_mode("classification") %>%
  fit(fmla, train_set)

#GENERATE IN-SAMPLE PREDICTIONS ON THE TRAIN SET AND COMBINE WITH TRAIN DATA
pred_class_in <- predict(default_tree, new_data = train_set, type="class") %>%
  bind_cols(train_set) #ADD CLASS PREDICTIONS DIRECTLY TO TEST DATA

#GENERATE IN-SAMPLE CONFUSION MATRIX AND DIAGNOSTICS
confusion_in <- table(pred_class_in$.pred_class, pred_class_in$Risk.Profile)
confusionMatrix(confusion_in) #FROM CARET PACKAGE

#GENERATE OUT-OF-SAMPLE PREDICTIONS ON THE TEST SET AND COMBINE WITH TEST DATA
pred_class_out <- predict(default_tree, new_data = test_set, type="class") %>%
  bind_cols(test_set) #ADD CLASS PREDICTIONS DIRECTLY TO TEST DATA

#GENERATE OUT-OF-SAMPLE CONFUSION MATRIX AND DIAGNOSTICS
confusion_out <- table(pred_class_out$.pred_class, pred_class_out$Risk.Profile)
confusionMatrix(confusion_out) #FROM CARET PACKAGE

#########################
##TUNING THE TREE MODEL##
#########################

#BLANK TREE SPECIFICATION FOR TUNING
tree_tune <- decision_tree(min_n = tune(),
                           tree_depth = tune(),
                           cost_complexity= tune()) %>%
  set_engine("rpart") %>%
  set_mode("classification")

#CREATING A TUNING PARAMETER GRID - USE ONE OF THESE AT A TIME
tree_tune_grid <- grid_regular(parameters(tree_tune), levels = 3)
#tree_tune_grid <- grid_random(parameters(tree_tune), size = 27) #FOR RANDOM GRID

#TUNING THE MODEL ALONG THE GRID W/ CROSS-VALIDATION
set.seed(123) #SET SEED FOR REPRODUCIBILITY WITH CROSS-VALIDATION
tree_tune_results <- tune_grid(tree_tune,
                          fmla, #MODEL FORMULA
                          resamples = vfold_cv(train_set, v=3), #RESAMPLES / FOLDS
                          grid = tree_tune_grid, #GRID
                          metrics = metric_set(accuracy)) #BENCHMARK METRIC

#RETRIEVE OPTIMAL PARAMETERS FROM CROSS-VALIDATION
best_params_tune <- select_best(tree_tune_results)

#FINALIZE THE MODEL SPECIFICATION
final_tune <- finalize_model(tree_tune, best_params_tune)

#FIT THE FINALIZED MODEL
final_model_tune <- final_tune %>% fit(fmla, train_set)

#GENERATE IN-SAMPLE PREDICTIONS ON THE TRAIN SET AND COMBINE WITH TRAIN DATA
pred_class_in_tune <- predict(final_model_tune, new_data = train_set, type="class") %>%
  bind_cols(train_set) #ADD CLASS PREDICTIONS DIRECTLY TO TEST DATA

#GENERATE IN-SAMPLE CONFUSION MATRIX AND DIAGNOSTICS
confusion_in_tune <- table(pred_class_in_tune$.pred_class, pred_class_in_tune$Risk.Profile)
confusionMatrix(confusion_in_tune) #FROM CARET PACKAGE

#GENERATE OUT-OF-SAMPLE PREDICTIONS ON THE TEST SET AND COMBINE WITH TEST DATA
pred_class_out_tune <- predict(final_model_tune, new_data = test_set, type="class") %>%
  bind_cols(test_set) #ADD CLASS PREDICTIONS DIRECTLY TO TEST DATA

#GENERATE OUT-OF-SAMPLE CONFUSION MATRIX AND DIAGNOSTICS
confusion_out_tune <- table(pred_class_out_tune$.pred_class, pred_class_out_tune$Risk.Profile)
confusionMatrix(confusion_out_tune) #FROM CARET PACKAGE

################################################################################

################################################################################
##This is the Bagging Forest Model.  ##
################################################################################

#LOADING THE LIBRARIES
library(dplyr)
library(tidymodels)
library(caret) #FOR confusionMatrix()
library(rpart.plot)
library(baguette) #FOR BAGGED TREES
library(xgboost) #FOR GRADIENT BOOSTING
library(vip) #FOR VARIABLE IMPORTANCE
install.packages("ranger")
install.packages("progress")
library(progress)
library(ranger)

#IMPORTING THE DATA
df <- read.csv("https://github.com/ahaywasUSD/576-Data-Group-Project/raw/main/data_synthetic_ATH.csv")
#CONVERT TO FACTORS
#df <- df %>%
#mutate_if(is.character, as.factor)
df$Risk.Profile<-as.factor(df$Risk.Profile) #CONVERT OUTPUT TO FACTOR

##PARTITIONING THE DATA##
# CREATE INDICES FOR DATA PARTITIONING
set.seed(123)
train_index <- createDataPartition(df$Risk.Profile, p = 0.7, list = FALSE, times = 1)
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

#MODEL DESCRIPTION:
fmla <- Risk.Profile ~ Premium.Amount + Credit.Score + Age

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
  fit(formula = fmla, data = train_set)
bagged_forest

###TUNING###
#Create spec w/ placeholders
bag_spec <- bag_tree(
  min_n = tune(),
  tree_depth = tune(),
  cost_complexity = tune()) %>%
  set_mode("classification") %>%
  set_engine("rpart")
#Create grid (regular or random)
tunegrid_bag <- grid_regular(parameters(bag_spec), levels = 3)
#tunegrid_bag <- grid_random(parameters(bag_spec), size = 27)
#Tune along grid
tune_results_bag <- tune_grid(
  bag_spec,
  Risk.Profile ~ Premium.Amount + Credit.Score + Age,
  resamples = vfold_cv(train_set, v = 3),
  grid = tunegrid_bag,
  metrics = metric_set(accuracy))
#Select final hyperparameters
best_params_bag <- select_best(tune_results_bag)
#Finalize spec
final_spec_bag <- finalize_model(bag_spec, best_params_bag)
final_spec_bag

#Train final model
final_bfmodel <- final_spec %>% fit(fmla, data = train_set)

#GENERATE IN-SAMPLE PREDICTIONS ON THE TRAIN SET AND COMBINE WITH TRAIN DATA
pred_class_in_bf <- predict(final_bfmodel, new_data = train_set, type="class") %>%
  bind_cols(train_set) #ADD CLASS PREDICTIONS DIRECTLY TO TEST DATA

#GENERATE IN-SAMPLE CONFUSION MATRIX AND DIAGNOSTICS
confusion_in_bf <- table(pred_class_in_bf$.pred_class, pred_class_in_bf$Risk.Profile)
confusionMatrix(confusion_in_bf) #FROM CARET PACKAGE

#GENERATE OUT-OF-SAMPLE PREDICTIONS ON THE TEST SET AND COMBINE WITH TEST DATA
pred_class_out_bf <- predict(final_bfmodel, new_data = test_set, type="class") %>%
  bind_cols(test_set) #ADD CLASS PREDICTIONS DIRECTLY TO TEST DATA

#GENERATE OUT-OF-SAMPLE CONFUSION MATRIX AND DIAGNOSTICS
confusion_out_bf <- table(pred_class_out_bf$.pred_class, pred_class_out_bf$Risk.Profile)
confusionMatrix(confusion_out_bf) #FROM CARET PACKAGE

################################################################################

################################################################################
##This is the Random Forest Model.  ##
################################################################################

#LOADING THE LIBRARIES
library(dplyr)
library(rpart.plot)
library(tidymodels)
library(baguette) #FOR BAGGED TREES
library(xgboost) #FOR GRADIENT BOOSTING
library(caret) #FOR confusionMatrix()
library(vip) #FOR VARIABLE IMPORTANCE
install.packages("ranger")
install.packages("progress")
library(progress)
library(ranger)
progress_bar

#IMPORTING THE DATA
df <- read.csv("https://github.com/ahaywasUSD/576-Data-Group-Project/raw/main/data_synthetic_ATH.csv")
#CONVERT TO FACTORS
#df <- df %>%
#mutate_if(is.character, as.factor)
df$Risk.Profile<-as.factor(df$Risk.Profile) #CONVERT OUTPUT TO FACTOR

##PARTITIONING THE DATA##
# CREATE INDICES FOR DATA PARTITIONING
set.seed(123)
train_index <- createDataPartition(df$Risk.Profile, p = 0.7, list = FALSE, times = 1)
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

#MODEL DESCRIPTION:
fmla <- Risk.Profile ~ Premium.Amount + Credit.Score + Age

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
  fit(formula = fmla, data = train_set) #%>%
print(random_forest)

#RANKING VARIABLE IMPORTANCE (CAN BE DONE WITH OTHER MODELS AS WELL)
set.seed(123) #NEED TO SET SEED WHEN FITTING OR BOOTSTRAPPED SAMPLES WILL CHANGE
rand_forest(min_n = 20 , #minimum number of observations for split
            trees = 100, #of ensemble members (trees in forest)
            mtry = 2)  %>% #number of variables to consider at each split
  set_mode("classification") %>%
  set_engine("ranger", importance = "impurity") %>%
  fit(fmla, data = train_set) %>%
  vip() #FROM VIP PACKAGE - ONLY WORKS ON RANGER FIT DIRECTLY

###TUNING###
#Create spec w/ placeholders
forest_spec <- rand_forest(
  min_n = tune(),
  trees = tune()) %>%
  set_mode("classification") %>%
  set_engine("ranger")
#Create grid (regular or random)
tunegrid_forest <- grid_regular(parameters(forest_spec), levels = 3)
#tunegrid_forest <- grid_random(parameters(forest_spec), size = 8)
#Tune along grid
tune_results_rf <- tune_grid(
  forest_spec,
  Risk.Profile ~ Premium.Amount + Credit.Score + Age,
  resamples = vfold_cv(train_set, v = 3),
  grid = tunegrid_forest,
  metrics = metric_set(accuracy))
#Select final hyperparameters
best_params_rf <- select_best(tune_results_rf)
#Finalize spec
final_spec_rf <- finalize_model(forest_spec, best_params_rf)
final_spec_rf
#Train final model
final_model_rf <- final_spec_rf %>% fit(fmla, data = train_set)

#GENERATE IN-SAMPLE PREDICTIONS ON THE TRAIN SET AND COMBINE WITH TRAIN DATA
pred_class_in_rf <- predict(final_model_rf, new_data = train_set, type="class") %>%
  bind_cols(train_set) #ADD CLASS PREDICTIONS DIRECTLY TO TEST DATA

#GENERATE IN-SAMPLE CONFUSION MATRIX AND DIAGNOSTICS
confusion_in_rf <- table(pred_class_in_rf$.pred_class, pred_class_in_rf$Risk.Profile)
confusionMatrix(confusion_in_rf) #FROM CARET PACKAGE

#GENERATE OUT-OF-SAMPLE PREDICTIONS ON THE TEST SET AND COMBINE WITH TEST DATA
pred_class_out_rf <- predict(final_model_rf, new_data = test_set, type="class") %>%
  bind_cols(test_set) #ADD CLASS PREDICTIONS DIRECTLY TO TEST DATA

#GENERATE OUT-OF-SAMPLE CONFUSION MATRIX AND DIAGNOSTICS
confusion_out_rf <- table(pred_class_out_rf$.pred_class, pred_class_out_rf$Risk.Profile)
confusionMatrix(confusion_out_rf) #FROM CARET PACKAGE

################################################################################

################################################################################
##This is the the Gradient-Boosted Model.  ##
################################################################################

#LOADING THE LIBRARIES
library(dplyr)
library(rpart.plot)
library(tidymodels)
library(baguette) #FOR BAGGED TREES
library(xgboost) #FOR GRADIENT BOOSTING
library(caret) #FOR confusionMatrix()
library(vip) #FOR VARIABLE IMPORTANCE
install.packages("ranger")
install.packages("progress")
library(progress)
library(ranger)
progress_bar

#IMPORTING THE DATA
df <- read.csv("https://github.com/ahaywasUSD/576-Data-Group-Project/raw/main/data_synthetic_ATH.csv")
#CONVERT TO FACTORS
#df <- df %>%
#mutate_if(is.character, as.factor)
df$Risk.Profile<-as.factor(df$Risk.Profile) #CONVERT OUTPUT TO FACTOR

##PARTITIONING THE DATA##
# CREATE INDICES FOR DATA PARTITIONING
set.seed(123)
train_index <- createDataPartition(df$Risk.Profile, p = 0.7, list = FALSE, times = 1)
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

#MODEL DESCRIPTION:
fmla <- Risk.Profile ~ Premium.Amount + Credit.Score + Age

###################################
#SPECIFYING GRADIENT BOOSTED MODEL#
###################################

#SPECIFY AND FIT IN ONE STEP:
boosted_forest <- boost_tree(min_n = NULL, #minimum number of observations for split
                             tree_depth = NULL, #max tree depth
                             trees = 100, #number of trees
                             mtry = NULL, #number of predictors selected at each split 
                             sample_size = NULL, #amount of data exposed to fitting
                             learn_rate = NULL, #learning rate for gradient descent
                             loss_reduction = NULL, #min loss reduction for further split
                             stop_iter = NULL)  %>% #maximum iteration for convergence
  set_engine("xgboost") %>%
  set_mode("classification") %>%
  fit(fmla, train_set)

###TUNING###
#Create spec w/ placeholders
boost_spec <- boost_tree(
  trees = 500,
  learn_rate = tune(),
  tree_depth = tune(),
  sample_size = tune()) %>%
  set_mode("classification") %>%
  set_engine("xgboost")
#Create grid (regular or random)
tunegrid_boost <- grid_regular(parameters(boost_spec), levels = 3)
#tunegrid_boost <- grid_random(parameters(boost_spec), size = 8)
#Tune along grid
tune_results_boost <- tune_grid(
  boost_spec,
  Risk.Profile ~ Premium.Amount + Credit.Score + Age,
  resamples = vfold_cv(train_set, v = 3),
  grid = tunegrid_boost,
  metrics = metric_set(accuracy))
#Select final hyperparameters
best_params_boost <- select_best(tune_results_boost)
#Finalize spec
final_spec_boost <- finalize_model(boost_spec, best_params_boost)
final_spec_boost
#Train final model
final_model_boost <- final_spec_boost %>% fit(fmla, data = train_set)

#GENERATE IN-SAMPLE PREDICTIONS ON THE TRAIN SET AND COMBINE WITH TRAIN DATA
pred_class_in_boost <- predict(final_model_boost, new_data = train_set, type="class") %>%
  bind_cols(train_set) #ADD CLASS PREDICTIONS DIRECTLY TO TEST DATA

#GENERATE IN-SAMPLE CONFUSION MATRIX AND DIAGNOSTICS
confusion_in_boost <- table(pred_class_in_boost$.pred_class, pred_class_in_boost$Risk.Profile)
confusionMatrix(confusion_in_boost) #FROM CARET PACKAGE

#GENERATE OUT-OF-SAMPLE PREDICTIONS ON THE TEST SET AND COMBINE WITH TEST DATA
pred_class_out_boost <- predict(final_model_boost, new_data = test_set, type="class") %>%
  bind_cols(test_set) #ADD CLASS PREDICTIONS DIRECTLY TO TEST DATA

#GENERATE OUT-OF-SAMPLE CONFUSION MATRIX AND DIAGNOSTICS
confusion_out_boost <- table(pred_class_out_boost$.pred_class, pred_class_out_boost$Risk.Profile)
confusionMatrix(confusion_out_boost) #FROM CARET PACKAGE

################################################################################

################################################################################
##This is the the Gradient-Boosted Model on the VALIDATION SET.  ##
################################################################################

#LOADING THE LIBRARIES
library(dplyr)
library(rpart.plot)
library(tidymodels)
library(baguette) #FOR BAGGED TREES
library(xgboost) #FOR GRADIENT BOOSTING
library(caret) #FOR confusionMatrix()
library(vip) #FOR VARIABLE IMPORTANCE
install.packages("ranger")
install.packages("progress")
library(progress)
library(ranger)
progress_bar

#IMPORTING THE DATA
df <- read.csv("https://github.com/ahaywasUSD/576-Data-Group-Project/raw/main/data_synthetic_ATH.csv")
#CONVERT TO FACTORS
#df <- df %>%
#mutate_if(is.character, as.factor)
df$Risk.Profile<-as.factor(df$Risk.Profile) #CONVERT OUTPUT TO FACTOR

##PARTITIONING THE DATA##
# CREATE INDICES FOR DATA PARTITIONING
set.seed(123)
train_index <- createDataPartition(df$Risk.Profile, p = 0.7, list = FALSE, times = 1)
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

#MODEL DESCRIPTION:
fmla <- Risk.Profile ~ Premium.Amount + Credit.Score + Age

###################################
#SPECIFYING GRADIENT BOOSTED MODEL#
###################################

#SPECIFY AND FIT IN ONE STEP:
boosted_forest <- boost_tree(min_n = NULL, #minimum number of observations for split
                             tree_depth = NULL, #max tree depth
                             trees = 100, #number of trees
                             mtry = NULL, #number of predictors selected at each split 
                             sample_size = NULL, #amount of data exposed to fitting
                             learn_rate = NULL, #learning rate for gradient descent
                             loss_reduction = NULL, #min loss reduction for further split
                             stop_iter = NULL)  %>% #maximum iteration for convergence
  set_engine("xgboost") %>%
  set_mode("classification") %>%
  fit(fmla, train_set)

###TUNING###
#Create spec w/ placeholders
boost_spec <- boost_tree(
  trees = 500,
  learn_rate = tune(),
  tree_depth = tune(),
  sample_size = tune()) %>%
  set_mode("classification") %>%
  set_engine("xgboost")
#Create grid (regular or random)
tunegrid_boost <- grid_regular(parameters(boost_spec), levels = 3)
#tunegrid_boost <- grid_random(parameters(boost_spec), size = 8)
#Tune along grid
tune_results_boost <- tune_grid(
  boost_spec,
  Risk.Profile ~ Premium.Amount + Credit.Score + Age,
  resamples = vfold_cv(train_set, v = 3),
  grid = tunegrid_boost,
  metrics = metric_set(accuracy))
#Select final hyperparameters
best_params_boost <- select_best(tune_results_boost)
#Finalize spec
final_spec_boost <- finalize_model(boost_spec, best_params_boost)
final_spec_boost
#Train final model
final_model_boost <- final_spec_boost %>% fit(fmla, data = train_set)

#GENERATE SAMPLE PREDICTIONS ON THE VALIDATION SET AND COMBINE WITH VALID DATA
pred_class_xb_valid <- predict(final_xbmodel, new_data = valid_set, type="class") %>%
  bind_cols(valid_set) #ADD CLASS PREDICTIONS DIRECTLY TO TEST DATA

#GENERATE SAMPLE CONFUSION MATRIX AND DIAGNOSTICS
confusion <- table(pred_class_xb_valid$.pred_class, pred_class_xb_valid$Risk.Profile)
confusionMatrix(confusion) #FROM CARET PACKAGE

################################################################################