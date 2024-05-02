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
tunegrid_forest <- grid_regular(parameters(forest_spec), levels = 5)
#tunegrid_forest <- grid_random(parameters(forest_spec), size = 8)
#Tune along grid
tune_results <- tune_grid(
  forest_spec,
  Risk.Profile ~ Premium.Amount + Credit.Score + Age,
  resamples = vfold_cv(train_set, v = 3),
  grid = tunegrid_forest,
  metrics = metric_set(accuracy))
#Select final hyperparameters
best_params <- select_best(tune_results)
#Finalize spec
final_spec <- finalize_model(spec_rf, best_params)
final_spec
#Train final model
final_model <- final_spec %>% fit(fmla, data = train_set)

#GENERATE IN-SAMPLE PREDICTIONS ON THE TRAIN SET AND COMBINE WITH TRAIN DATA
pred_class_rf_in <- predict(random_forest, new_data = train_set, type="class") %>%
  bind_cols(train_set) #ADD CLASS PREDICTIONS DIRECTLY TO TEST DATA

#GENERATE IN-SAMPLE CONFUSION MATRIX AND DIAGNOSTICS
confusion <- table(pred_class_rf_in$.pred_class, pred_class_rf_in$Risk.Profile)
confusionMatrix(confusion) #FROM CARET PACKAGE

#GENERATE OUT-OF-SAMPLE PREDICTIONS ON THE TEST SET AND COMBINE WITH TEST DATA
pred_class_rf_out <- predict(random_forest, new_data = test_set, type="class") %>%
  bind_cols(test_set) #ADD CLASS PREDICTIONS DIRECTLY TO TEST DATA

#GENERATE OUT-OF-SAMPLE CONFUSION MATRIX AND DIAGNOSTICS
confusion <- table(pred_class_rf_out$.pred_class, pred_class_rf_out$Risk.Profile)
confusionMatrix(confusion) #FROM CARET PACKAGE

################################################################################