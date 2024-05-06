# IMPORTING THE DATA
df <- read.csv("https://github.com/ahaywasUSD/576-Data-Group-Project/raw/main/data_synthetic_ATH.csv")
df$Claim.History = ifelse(df$Claim.History == 0, 0, 1)
df$Claim.History = as.factor(df$Claim.History)

df <- df %>%
  mutate_if(is.character, as.factor)

library(ggplot2)
library(rsample) #randomized stratified sampling
library(caret)
library(e1071)

#Claim History graph
ggplot(data = df, aes(x = Claim.History)) +
  geom_bar(fill = "orange", alpha = 0.6) +
  labs(title = "Distribution of Claim History (Binary)",
       x = "Category",
       y = "Frequency") +
  theme_minimal()

##PARTITIONING THE DATA##
# CREATE INDICES FOR DATA PARTITIONING
set.seed(123)
split<-initial_split(df, .7, strata=Risk.Profile) #stratifying by Risk.Profile to ensure same in and out of 
train_set<-training(split)
holdout_set <- testing(split)
split2<-initial_split(holdout_set, .5, strata=Risk.Profile) 
test_set <- testing(split2)
valid_set <- training(split2) #using the training function to create validation set, kind of counterintuitive but gives use the desired end result

# SAVE PARTITIONS AS CSV
write.csv(train_set, "train_set.csv", row.names = FALSE)
write.csv(valid_set, "valid_set.csv", row.names = FALSE)
write.csv(test_set, "test_set.csv", row.names = FALSE)

#a) Logistic regression model 
m0 = glm(Claim.History ~ ., train_set, family = binomial(link = "logit")) # model with all variables
summary(m0)

fmla <- Claim.History ~ . - (Age + Occupation + Location + Education.Level + Premium.Amount + Income.Level + Deductible + Policy.Type + Driving.Record + Previous.Claims.History)

m1 = glm(fmla, data = train_set, family = binomial(link = "logit"))
summary(m1)

confint <-exp(confint(m1))

#b) Probit model

m2 = glm(fmla, data = train_set, family = binomial(link ="probit"))
summary(m2)

confint <-exp(confint(m2))


#Confusion tables

#Logit
#In-sample
pred <- predict(m1, train_set, type = "response")
confusionMatrix(table(pred >= .75, train_set$Claim.History == 1), positive='TRUE')
#Out-of-sample
pred_out <- predict(m1, test_set, type = "response")
confusionMatrix(table(pred_out>=0.75, test_set$Claim.History ==1), positve = "TRUE")

#Probit
#In-sample
pred_probit <- predict(m2, train_set, type = "response")
confusion_probit<-table(pred_probit >0.75, train_set$Claim.History == 1)
confusionMatrix(confusion_probit, positive='TRUE')

#Out-of-sample
pred_probit_out <- predict(m2, test_set, type = "response")
confusionMatrix(table(pred_probit_out>0.75, test_set$Claim.History==1), positive='TRUE')

#ROC
lixbrary(pROC)

#Logit
pva<-data.frame(preds=pred, actual=factor(train_set$Claim.History))
roc_obj <- roc(pva$actual, pva$preds) #NB this line displays an error message but following computations work

#NOTE THIS PLOTS SENSITIVITY (TRUE POSITIVES) VS. SPECIFICITY (TRUE NEGATIVES)
plot(roc_obj, col='blue', main="ROC Curve")

(auc <- 1-auc(roc_obj))

#Probit
pva<-data.frame(preds=pred_probit, actual=factor(train_set$Claim.History))
roc_obj <- roc(pva$actual, pva$preds) 

#NOTE THIS PLOTS SENSITIVITY (TRUE POSITIVES) VS. SPECIFICITY (TRUE NEGATIVES)
plot(roc_obj, col='blue', main="ROC Curve")
(auc <- 1-auc(roc_obj))

