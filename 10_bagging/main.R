# Helper packages
library(dplyr)       # for data wrangling
library(ggplot2)     # for awesome plotting
library(doParallel)  # for parallel backend to foreach
library(foreach)     # for parallel processing with for loops

# Modeling packages
library(caret)       # for general model fitting
library(rpart)       # for fitting decision trees
library(ipred)       # for fitting bagged decision trees

ames <- AmesHousing::make_ames()

set.seed(123)
split <- rsample::initial_split(ames, prop=0.7, strata="Sale_Price")
ames_train <- rsample::training(split)
ames_test <- rsample::testing(split)

### Using ipred package

# make bootstrapping reproducible
set.seed(123)

# train bagged model
ames_bag1 <- bagging(
  formula = Sale_Price ~ .,
  data = ames_train,
  nbagg = 100,  
  coob = TRUE,
  control = rpart.control(minsplit = 2, cp = 0)
)

ames_bag1
## 
## Bagging regression trees with 100 bootstrap replications 
## 
## Call: bagging.data.frame(formula = Sale_Price ~ ., data = ames_train, 
##     nbagg = 100, coob = TRUE, control = rpart.control(minsplit = 2, 
##         cp = 0))
## 
## Out-of-bag estimate of root mean squared error:  25528.78

### Using caret package
ames_bag2 <- train(
  Sale_Price ~ .,
  data = ames_train,
  method = "treebag",
  trControl = trainControl(method = "cv", number = 10),
  nbagg = 200,  
  control = rpart.control(minsplit = 2, cp = 0)
)
ames_bag2
## Bagged CART 
## 
## 2054 samples
##   80 predictor
## 
## No pre-processing
## Resampling: Cross-Validated (10 fold) 
## Summary of sample sizes: 1849, 1848, 1848, 1849, 1849, 1847, ... 
## Resampling results:
## 
##   RMSE      Rsquared   MAE     
##   26957.06  0.8900689  16713.14

### Parallel processing
# Create a parallel socket cluster
cl <- makeCluster(8) # use 8 workers
registerDoParallel(cl) # register the parallel backend

# Fit trees in parallel and compute predictions on the test set
predictions <- foreach(
  icount(160), 
  .packages = "rpart", 
  .combine = cbind
) %dopar% {
  # bootstrap copy of training data
  index <- sample(nrow(ames_train), replace = TRUE)
  ames_train_boot <- ames_train[index, ]  
  
  # fit tree to bootstrap copy
  bagged_tree <- rpart(
    Sale_Price ~ ., 
    control = rpart.control(minsplit = 2, cp = 0),
    data = ames_train_boot
  ) 
  
  predict(bagged_tree, newdata = ames_test)
}

predictions[1:5, 1:7]
##   result.1 result.2 result.3 result.4 result.5 result.6 result.7
## 1   176500   187000   179900   187500   187500   187500   187500
## 2   180000   254000   251000   240000   180000   180000   221000
## 3   175000   174000   192000   192000   185000   178900   163990
## 4   197900   157000   217500   215000   180000   210000   218500
## 5   120000   129000   130000   143000   136500   153600   148500

#plot RMSE vs #trees
predictions %>%
  as.data.frame() %>%
  mutate(
    observation = 1:n(),
    actual = ames_test$Sale_Price) %>%
  tidyr::gather(tree, predicted, -c(observation, actual)) %>%
  group_by(observation) %>%
  mutate(tree = stringr::str_extract(tree, '\\d+') %>% as.numeric()) %>%
  ungroup() %>%
  arrange(observation, tree) %>%
  group_by(observation) %>%
  mutate(avg_prediction = cummean(predicted)) %>%
  group_by(tree) %>%
  summarize(RMSE = RMSE(avg_prediction, actual)) %>%
  ggplot(aes(tree, RMSE)) +
  geom_line() +
  xlab('Number of trees')

# Shutdown parallel cluster
stopCluster(cl)

### Feature interpretation (only for caret's output)
vip::vip(ames_bag2, num_features = 40, bar = FALSE)

# Construct partial dependence plots
p1 <- pdp::partial(
  ames_bag2, 
  pred.var = "Lot_Area",
  grid.resolution = 20
) %>% 
  autoplot()

p2 <- pdp::partial(
  ames_bag2, 
  pred.var = "Lot_Frontage", 
  grid.resolution = 20
) %>% 
  autoplot()

gridExtra::grid.arrange(p1, p2, nrow = 1)