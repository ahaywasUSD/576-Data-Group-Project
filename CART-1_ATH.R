# LOAD THE LIBRARIES
library(dplyr)
library(tidymodels) #INCLUDES parsnip PACKAGE FOR decision_tree()
library(caret) #FOR confusionMatrix()
library(rpart.plot)

# IMPORTING THE DATA
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

# SPECIFYING THE CLASSIFICATION TREE MODEL
class_spec <- decision_tree(min_n = 20, #minimum number of observations for split
                            tree_depth = 30, #max tree depth
                            cost_complexity = 0.01)  %>% #regularization parameter
  set_engine("rpart") %>%
  set_mode("classification")
print(class_spec)

#ESTIMATING THE MODEL (CAN BE DONE IN ONE STEP ABOVE WITH EXTRA %>%)
class_fmla <- Education.Level ~ .
class_tree <- class_spec %>%
  fit(formula = class_fmla, data = train_set)
print(class_tree)

#VISUALIZING THE CLASSIFICATION TREE MODEL:
class_tree$fit %>%
  rpart.plot(type = 1, extra = 2, roundint = FALSE)

plotcp(class_tree$fit)

#GENERATE IN-SAMPLE PREDICTIONS ON THE TRAIN SET AND COMBINE WITH TRAIN DATA
pred_class_in <- predict(class_tree, new_data = train_set, type="class") %>%
  bind_cols(train_set) #ADD CLASS PREDICTIONS DIRECTLY TO TEST DATA

#GENERATE IN-SAMPLE CONFUSION MATRIX AND DIAGNOSTICS
confusion <- table(pred_class_in$.pred_class, pred_class_in$Education.Level)
confusionMatrix(confusion) #FROM CARET PACKAGE

#GENERATE OUT-OF-SAMPLE PREDICTIONS ON THE TEST SET AND COMBINE WITH TEST DATA
pred_class_out <- predict(class_tree, new_data = test_set, type="class") %>%
  bind_cols(test_set) #ADD CLASS PREDICTIONS DIRECTLY TO TEST DATA

#GENERATE OUT-OF-SAMPLE CONFUSION MATRIX AND DIAGNOSTICS
confusion <- table(pred_class_out$.pred_class, pred_class_out$Education.Level)
confusionMatrix(confusion) #FROM CARET PACKAGE

##############

#SPECIFYING THE REGRESSION TREE MODEL
reg_spec <- decision_tree(min_n = 20 , #minimum number of observations for split
                          tree_depth = 30, #max tree depth
                          cost_complexity = 0.01)  %>% #regularization parameter
  set_engine("rpart") %>%
  set_mode("regression")
print(reg_spec)

#ESTIMATING THE MODEL (CAN BE DONE IN ONE STEP ABOVE WITH EXTRA %>%)
reg_fmla <- Credit.Score ~ .
reg_tree <- reg_spec %>%
  fit(formula = reg_fmla, data = train_set)
print(reg_tree)

#VISUALIZING THE REGRESSION TREE
reg_tree$fit %>%
  rpart.plot(type = 2, roundint = FALSE)

#GENERATE PREDICTIONS AND COMBINE WITH TEST SET
pred_reg <- predict(reg_tree, new_data = test_set) %>%
  bind_cols(test_set)

#OUT-OF-SAMPLE ERROR ESTIMATES FROM yardstick OR ModelMetrics PACKAGE
mae(pred_reg, estimate=.pred, truth=Credit.Score) 
rmse(pred_reg, estimate=.pred, truth=Credit.Score)
