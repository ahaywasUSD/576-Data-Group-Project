################################################################################
##This is the Bagged Tree Model.  ##
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
#tunegrid_bag <- grid_regular(parameters(bag_spec), levels = 3)
tunegrid_bag <- grid_random(parameters(bag_spec), size = 27)
#Tune along grid
tune_results <- tune_grid(
  bag_spec,
  Risk.Profile ~ Premium.Amount + Credit.Score + Age,
  resamples = vfold_cv(train_set, v = 3),
  grid = tunegrid_bag,
  metrics = metric_set(accuracy))
#Select final hyperparameters
best_params <- select_best(tune_results)
#Finalize spec
final_spec <- finalize_model(bag_spec, best_params)
final_spec
#Train final model
final_model <- final_spec %>% fit(fmla, data = train_set)

#GENERATE IN-SAMPLE PREDICTIONS ON THE TRAIN SET AND COMBINE WITH TRAIN DATA
pred_class_bf_in <- predict(bagged_forest, new_data = train_set, type="class") %>%
  bind_cols(train_set) #ADD CLASS PREDICTIONS DIRECTLY TO TEST DATA

#GENERATE IN-SAMPLE CONFUSION MATRIX AND DIAGNOSTICS
confusion <- table(pred_class_bf_in$.pred_class, pred_class_bf_in$Risk.Profile)
confusionMatrix(confusion) #FROM CARET PACKAGE

#GENERATE OUT-OF-SAMPLE PREDICTIONS ON THE TEST SET AND COMBINE WITH TEST DATA
pred_class_bf_out <- predict(bagged_forest, new_data = test_set, type="class") %>%
  bind_cols(test_set) #ADD CLASS PREDICTIONS DIRECTLY TO TEST DATA

#GENERATE OUT-OF-SAMPLE CONFUSION MATRIX AND DIAGNOSTICS
confusion <- table(pred_class_bf_out$.pred_class, pred_class_bf_out$Risk.Profile)
confusionMatrix(confusion) #FROM CARET PACKAGE

################################################################################