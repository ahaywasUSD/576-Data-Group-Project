#LOADING THE LIBRARIES
library(dplyr)
library(tidymodels) #INCLUDES parsnip PACKAGE FOR decision_tree()
library(caret) #FOR confusionMatrix()
library(rpart.plot)

#IMPORTING THE DATA
df <- read.csv("https://github.com/ahaywasUSD/576-Data-Group-Project/raw/main/data_synthetic_ATH.csv")
#CONVERT TO FACTORS
df <- df %>%
  mutate_if(is.character, as.factor)

##PARTITIONING THE DATA##
# CREATE INDICES FOR DATA PARTITIONING
set.seed(123)
train_index <- createDataPartition(df$Education.Level, p = 0.7, list = FALSE, times = 1)
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
fmla <- Education.Level ~ .

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
confusion <- table(pred_class_in$.pred_class, pred_class_in$Education.Level)
confusionMatrix(confusion) #FROM CARET PACKAGE

#GENERATE OUT-OF-SAMPLE PREDICTIONS ON THE TEST SET AND COMBINE WITH TEST DATA
pred_class_out <- predict(default_tree, new_data = test_set, type="class") %>%
  bind_cols(test_set) #ADD CLASS PREDICTIONS DIRECTLY TO TEST DATA

#GENERATE OUT-OF-SAMPLE CONFUSION MATRIX AND DIAGNOSTICS
confusion <- table(pred_class_out$.pred_class, pred_class_out$Education.Level)
confusionMatrix(confusion) #FROM CARET PACKAGE

#########################
##TUNING THE TREE MODEL##
#########################

#BLANK TREE SPECIFICATION FOR TUNING
tree_spec <- decision_tree(min_n = tune(),
                           tree_depth = tune(),
                           cost_complexity= tune()) %>%
  set_engine("rpart") %>%
  set_mode("classification")

#CREATING A TUNING PARAMETER GRID
tree_grid <- grid_regular(parameters(tree_spec), levels = 3)
#tree_grid <- grid_random(parameters(tree_spec), size = 3) FOR RANDOM GRID

#TUNING THE MODEL ALONG THE GRID W/ CROSS-VALIDATION
set.seed(123) #SET SEED FOR REPRODUCIBILITY WITH CROSS-VALIDATION
tune_results <- tune_grid(tree_spec,
                          fmla, #MODEL FORMULA
                          resamples = vfold_cv(train_set, v=3), #RESAMPLES / FOLDS
                          grid = tree_grid, #GRID
                          metrics = metric_set(accuracy)) #BENCHMARK METRIC

#RETRIEVE OPTIMAL PARAMETERS FROM CROSS-VALIDATION
best_params <- select_best(tune_results)

#FINALIZE THE MODEL SPECIFICATION
final_spec <- finalize_model(tree_spec, best_params)

#FIT THE FINALIZED MODEL
final_model <- final_spec %>% fit(fmla, train_set)

#GENERATE IN-SAMPLE PREDICTIONS ON THE TRAIN SET AND COMBINE WITH TRAIN DATA
pred_class_in <- predict(final_model, new_data = train_set, type="class") %>%
  bind_cols(train_set) #ADD CLASS PREDICTIONS DIRECTLY TO TEST DATA

#GENERATE IN-SAMPLE CONFUSION MATRIX AND DIAGNOSTICS
confusion <- table(pred_class_in$.pred_class, pred_class_in$Education.Level)
confusionMatrix(confusion) #FROM CARET PACKAGE

#GENERATE OUT-OF-SAMPLE PREDICTIONS ON THE TEST SET AND COMBINE WITH TEST DATA
pred_class_out <- predict(final_model, new_data = test_set, type="class") %>%
  bind_cols(test_set) #ADD CLASS PREDICTIONS DIRECTLY TO TEST DATA

#GENERATE OUT-OF-SAMPLE CONFUSION MATRIX AND DIAGNOSTICS
confusion <- table(pred_class_out$.pred_class, pred_class_out$Education.Level)
confusionMatrix(confusion) #FROM CARET PACKAGE
