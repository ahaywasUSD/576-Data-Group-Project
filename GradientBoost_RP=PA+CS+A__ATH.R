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
tune_results <- tune_grid(
  boost_spec,
  Risk.Profile ~ Premium.Amount + Credit.Score + Age,
  resamples = vfold_cv(train_set, v = 3),
  grid = tunegrid_boost,
  metrics = metric_set(accuracy))
#Select final hyperparameters
best_params <- select_best(tune_results)
#Finalize spec
final_spec <- finalize_model(boost_spec, best_params)
final_spec
#Train final model
final_model <- final_spec %>% fit(fmla, data = train_set)

#GENERATE IN-SAMPLE PREDICTIONS ON THE TRAIN SET AND COMBINE WITH TRAIN DATA
pred_class_xb_in <- predict(boosted_forest, new_data = train_set, type="class") %>%
  bind_cols(train_set) #ADD CLASS PREDICTIONS DIRECTLY TO TEST DATA

#GENERATE IN-SAMPLE CONFUSION MATRIX AND DIAGNOSTICS
confusion <- table(pred_class_xb_in$.pred_class, pred_class_xb_in$Risk.Profile)
confusionMatrix(confusion) #FROM CARET PACKAGE

#GENERATE OUT-OF-SAMPLE PREDICTIONS ON THE TEST SET AND COMBINE WITH TEST DATA
pred_class_xb_out <- predict(boosted_forest, new_data = test_set, type="class") %>%
  bind_cols(test_set) #ADD CLASS PREDICTIONS DIRECTLY TO TEST DATA

#GENERATE OUT-OF-SAMPLE CONFUSION MATRIX AND DIAGNOSTICS
confusion <- table(pred_class_xb_out$.pred_class, pred_class_xb_out$Risk.Profile)
confusionMatrix(confusion) #FROM CARET PACKAGE

################################################################################