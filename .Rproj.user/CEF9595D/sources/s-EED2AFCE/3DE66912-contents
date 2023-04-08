# Helper packages
library(recipes)  # for feature engineering

# Modeling packages
library(glmnet)   # for implementing regularized regression
library(caret)    # for automating the tuning process

# Model interpretability packages
library(vip)      # for variable importance

ames <- AmesHousing::make_ames()

set.seed(123)
split <- rsample::initial_split(ames, prop=0.7, strata="Sale_Price")
ames_train <- rsample::training(split)
ames_test <- rsample::testing(split)

# Create training  feature matrices
# we use model.matrix(...)[, -1] to discard the intercept
X <- model.matrix(Sale_Price ~ ., ames_train)[, -1]

# transform y with log transformation
Y <- log(ames_train$Sale_Price)

### Ridge
# Apply ridge regression to ames data
ridge <- glmnet(
  x = X,
  y = Y,
  alpha = 0
)

plot(ridge, xvar = "lambda")

### Cross validation 
# Apply CV ridge regression to Ames data
ridge <- cv.glmnet(
  x = X,
  y = Y,
  alpha = 0
)

# Apply CV lasso regression to Ames data
lasso <- cv.glmnet(
  x = X,
  y = Y,
  alpha = 1
)

# plot results
par(mfrow = c(1, 2))
plot(ridge, main = "Ridge penalty\n\n")
plot(lasso, main = "Lasso penalty\n\n")

### Using caret package to find the best lambda (how much penalty) and alpha (alpha=0 --> Ridge, alpha=1 --> Lasso, betweeon 0 and 1 --> ENET)
# grid search across 
cv_glmnet <- train(
  x = X,
  y = Y,
  method = "glmnet",
  preProc = c("zv", "center", "scale"),
  trControl = trainControl(method = "cv", number = 10),
  tuneLength = 10
)

# model with lowest RMSE
cv_glmnet$bestTune
##   alpha     lambda
## 7   0.1 0.02007035

# results for model with lowest RMSE
cv_glmnet$results %>%
  filter(alpha == cv_glmnet$bestTune$alpha, lambda == cv_glmnet$bestTune$lambda)
##   alpha     lambda      RMSE  Rsquared        MAE     RMSESD RsquaredSD
## 1   0.1 0.02007035 0.1277585 0.9001487 0.08102427 0.02235901  0.0346677
##         MAESD
## 1 0.005667366

# plot cross-validated RMSE
ggplot(cv_glmnet)

### Regularized logitstic regression
df <- attrition %>% mutate_if(is.ordered, factor, ordered = FALSE)

# Create training (70%) and test (30%) sets for the
# rsample::attrition data. Use set.seed for reproducibility
set.seed(123)
churn_split <- initial_split(df, prop = .7, strata = "Attrition")
train <- training(churn_split)
test  <- testing(churn_split)

# train logistic regression model
set.seed(123)
glm_mod <- train(
  Attrition ~ ., 
  data = train, 
  method = "glm",
  family = "binomial",
  preProc = c("zv", "center", "scale"),
  trControl = trainControl(method = "cv", number = 10)
)

# train regularized logistic regression model
set.seed(123)
penalized_mod <- train(
  Attrition ~ ., 
  data = train, 
  method = "glmnet",
  family = "binomial",
  preProc = c("zv", "center", "scale"),
  trControl = trainControl(method = "cv", number = 10),
  tuneLength = 10
)

# extract out of sample performance measures
summary(resamples(list(
  logistic_model = glm_mod, 
  penalized_model = penalized_mod
)))$statistics$Accuracy
##                      Min.   1st Qu.    Median      Mean   3rd Qu.
## logistic_model  0.8365385 0.8495146 0.8792476 0.8757893 0.8907767
## penalized_model 0.8446602 0.8759280 0.8834951 0.8835759 0.8915469
##                      Max. NA's
## logistic_model  0.9313725    0
## penalized_model 0.9411765    0