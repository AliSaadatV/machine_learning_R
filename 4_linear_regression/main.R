# Helper packages
library(dplyr)    # for data manipulation
library(ggplot2)  # for awesome graphics

# Modeling packages
library(caret)    # for cross-validation, etc.

# Model interpretability packages
library(vip)      # variable importance

ames <- AmesHousing::make_ames()
split <- rsample::initial_split(ames, prop=0.7, strata="Sale_Price")
ames_train <- rsample::training(split)
ames_test <- rsample::testing(split)

### Ordinary linear regression
set.seed(123)
cv_model_lm <- train(
  Sale_Price ~ ., 
  data = ames_train, 
  method = "lm",
  trControl = trainControl(method = "cv", number = 10)
)

### Principal component regression (PCR)
set.seed(123)
cv_model_pcr <- train(
  Sale_Price ~ ., 
  data = ames_train, 
  method = "pcr",
  trControl = trainControl(method = "cv", number = 10),
  preProcess = c("zv", "center", "scale"),
  tuneLength = 100
)

### Partial least squares (PLS): it is similar to PCR, but it uses y values to for PC construction
set.seed(123)
cv_model_pls <- train(
  Sale_Price ~ ., 
  data = ames_train, 
  method = "pls",
  trControl = trainControl(method = "cv", number = 10),
  preProcess = c("zv", "center", "scale"),
  tuneLength = 30
)

### Interpretation: variable importance (vip) and partial dependance plots (pdp)
vip(cv_model_pls, num_features = 20, method = "model") #this shows the most imoortant features

pdp::partial(cv_model_pls, "Gr_Liv_Area", grid.resolution = 20, plot = TRUE) #shows relationship between predicted outcome and chosen feature
