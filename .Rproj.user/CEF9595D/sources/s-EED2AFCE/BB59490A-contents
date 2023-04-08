# Helper packages
library(dplyr)     # for data wrangling
library(ggplot2)   # for awesome plotting
library(rsample)   # for data splitting

# Modeling packages
library(caret)     # for logistic regression modeling

# Model interpretability packages
library(vip)       # variable importance

churn <- modeldata::attrition %>%  mutate_if(is.ordered, .funs = factor, ordered = FALSE)

# Create training (70%) and test (30%) sets for the 
# rsample::attrition data.
set.seed(123)  # for reproducibility
churn_split <- initial_split(churn, prop = .7, strata = "Attrition")
churn_train <- training(churn_split)
churn_test  <- testing(churn_split)

### logistic regression
set.seed(123)
cv_model1 <- train(
  Attrition ~ MonthlyIncome, 
  data = churn_train, 
  method = "glm",
  family = "binomial",
  trControl = trainControl(method = "cv", number = 10)
)

set.seed(123)
cv_model2 <- train(
  Attrition ~ MonthlyIncome + OverTime, 
  data = churn_train, 
  method = "glm",
  family = "binomial",
  trControl = trainControl(method = "cv", number = 10)
)

set.seed(123)
cv_model3 <- train(
  Attrition ~ ., 
  data = churn_train, 
  method = "glm",
  family = "binomial",
  trControl = trainControl(method = "cv", number = 10)
)

# extract out of sample performance measures
summary(
  resamples(
    list(
      model1 = cv_model1, 
      model2 = cv_model2, 
      model3 = cv_model3
    )
  )
)$statistics$Accuracy
##             Min.   1st Qu.    Median      Mean   3rd Qu.      Max. NA's
## model1 0.8349515 0.8349515 0.8365385 0.8388478 0.8431373 0.8446602    0
## model2 0.8349515 0.8349515 0.8365385 0.8388478 0.8431373 0.8446602    0
## model3 0.8365385 0.8495146 0.8792476 0.8757893 0.8907767 0.9313725    0

### Confusion matrix
# predict class
pred_class <- predict(cv_model3, churn_train)

# create confusion matrix
confusionMatrix(
  data = relevel(pred_class, ref = "Yes"), 
  reference = relevel(churn_train$Attrition, ref = "Yes")
)

### ROC analysis
library(ROCR)

# Compute predicted probabilities
m1_prob <- predict(cv_model1, churn_train, type = "prob")$Yes
m2_prob <- predict(cv_model2, churn_train, type = "prob")$Yes
m3_prob <- predict(cv_model3, churn_train, type = "prob")$Yes

# Compute AUC metrics for cv_model1 and cv_model3
perf1 <- prediction(m1_prob, churn_train$Attrition) %>%
  performance(measure = "tpr", x.measure = "fpr")
perf2 <- prediction(m2_prob, churn_train$Attrition) %>% 
  performance(measure = "tpr", x.measure = "fpr")
perf3 <- prediction(m3_prob, churn_train$Attrition) %>%
  performance(measure = "tpr", x.measure = "fpr")

plot(perf1, col = "black", lty = 2)
plot(perf2, add = TRUE, col = "blue")
plot(perf3, add = TRUE, col = "gray")
legend(0.8, 0.2, legend = c("cv_model1", "cv_model2", "cv_model3"),
       col = c("black", "blue", "gray"), lty = 2:1, cex = 0.6)


### Partial least squares (reduces number of numeric features, then performs logistic regression)
# Perform 10-fold CV on a PLS model tuning the number of PCs to 
# use as predictors
set.seed(123)
cv_model_pls <- train(
  Attrition ~ ., 
  data = churn_train, 
  method = "pls",
  family = "binomial",
  trControl = trainControl(method = "cv", number = 10),
  preProcess = c("zv", "center", "scale"),
  tuneLength = 16
)

# Model with lowest RMSE
cv_model_pls$bestTune
##    ncomp
## 14    14

# results for model with lowest loss
cv_model_pls$results %>%
  dplyr::filter(ncomp == pull(cv_model_pls$bestTune))
##   ncomp  Accuracy     Kappa AccuracySD   KappaSD
## 1    14 0.8757518 0.3766944 0.01919777 0.1142592

# Plot cross-validated RMSE
ggplot(cv_model_pls)

### Interpretation
vip(cv_model3, num_features = 20)
