# Helper packages
library(dplyr)      # for data wrangling
library(ggplot2)    # for awesome graphics
library(rsample)    # for creating validation splits
library(recipes)    # for feature engineering

# Modeling packages
library(caret)       # for fitting KNN models

ames <- AmesHousing::make_ames()

set.seed(123)
split <- rsample::initial_split(ames, prop=0.7, strata="Sale_Price")
ames_train <- rsample::training(split)
ames_test <- rsample::testing(split)

# create training (70%) set for the rsample::attrition data.
attrit <- modeldata::attrition %>% mutate_if(is.ordered, factor, ordered = FALSE)
set.seed(123)
churn_split <- initial_split(attrit, prop = .7, strata = "Attrition")
churn_train <- training(churn_split)

# Create blueprint
blueprint <- recipe(Attrition ~ ., data = churn_train) %>%
  step_nzv(all_nominal()) %>%
  step_integer(contains("Satisfaction")) %>%
  step_integer(WorkLifeBalance) %>%
  step_integer(JobInvolvement) %>%
  step_dummy(all_nominal(), -all_outcomes(), one_hot = TRUE) %>%
  step_center(all_numeric(), -all_outcomes()) %>%
  step_scale(all_numeric(), -all_outcomes())

# Create a resampling method
cv <- trainControl(
  method = "repeatedcv", 
  number = 10, 
  repeats = 5,
  classProbs = TRUE,                 
  summaryFunction = twoClassSummary
)

# Create a hyperparameter grid search
hyper_grid <- expand.grid(
  k = floor(seq(1, nrow(churn_train)/3, length.out = 20))
)

# Fit knn model and perform grid search
knn_grid <- train(
  blueprint, 
  data = churn_train, 
  method = "knn", 
  trControl = cv, 
  tuneGrid = hyper_grid,
  metric = "ROC"
)

ggplot(knn_grid)
