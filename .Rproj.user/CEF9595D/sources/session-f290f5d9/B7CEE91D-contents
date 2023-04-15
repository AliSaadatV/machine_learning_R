# Helper packages
library(dplyr)    # for general data wrangling needs

# Modeling packages
library(gbm)      # for original implementation of regular and stochastic GBMs
library(h2o)      # for a java-based implementation of GBM variants
library(xgboost)  # for fitting extreme gradient boosting

ames <- AmesHousing::make_ames()

set.seed(123)
split <- rsample::initial_split(ames, prop=0.7, strata="Sale_Price")
ames_train <- rsample::training(split)
ames_test <- rsample::testing(split)

h2o.init(max_mem_size = "10g")

train_h2o <- as.h2o(ames_train)
response <- "Sale_Price"
predictors <- setdiff(colnames(ames_train), response)

### Basic GBM

# run a basic GBM model
set.seed(123)  # for reproducibility
ames_gbm1 <- gbm(
  formula = Sale_Price ~ .,
  data = ames_train,
  distribution = "gaussian",  # SSE loss function
  n.trees = 5000,
  shrinkage = 0.1,
  interaction.depth = 3,
  n.minobsinnode = 10,
  cv.folds = 10
)

# find index for number trees with minimum CV error
best <- which.min(ames_gbm1$cv.error)

# get MSE and compute RMSE
sqrt(ames_gbm1$cv.error[best])
## [1] 23240.38

# plot error curve
gbm.perf(ames_gbm1, method = "cv")


# create grid search
hyper_grid <- expand.grid(
  learning_rate = c(0.3, 0.1, 0.05, 0.01, 0.005),
  RMSE = NA,
  trees = NA,
  time = NA
)

# execute grid search
for(i in seq_len(nrow(hyper_grid))) {
  
  # fit gbm
  set.seed(123)  # for reproducibility
  train_time <- system.time({
    m <- gbm(
      formula = Sale_Price ~ .,
      data = ames_train,
      distribution = "gaussian",
      n.trees = 5000, 
      shrinkage = hyper_grid$learning_rate[i], 
      interaction.depth = 3, 
      n.minobsinnode = 10,
      cv.folds = 10 
    )
  })
  
  # add SSE, trees, and training time to results
  hyper_grid$RMSE[i]  <- sqrt(min(m$cv.error))
  hyper_grid$trees[i] <- which.min(m$cv.error)
  hyper_grid$Time[i]  <- train_time[["elapsed"]]
  
}

# results
arrange(hyper_grid, RMSE)
##   learning_rate  RMSE trees  time
## 1         0.050 21382  2375 129.5
## 2         0.010 21828  4982 126.0
## 3         0.100 22252   874 137.6
## 4         0.005 23136  5000 136.8
## 5         0.300 24454   427 139.9

# search grid
hyper_grid <- expand.grid(
  n.trees = 6000,
  shrinkage = 0.01,
  interaction.depth = c(3, 5, 7),
  n.minobsinnode = c(5, 10, 15)
)

# create model fit function
model_fit <- function(n.trees, shrinkage, interaction.depth, n.minobsinnode) {
  set.seed(123)
  m <- gbm(
    formula = Sale_Price ~ .,
    data = ames_train,
    distribution = "gaussian",
    n.trees = n.trees,
    shrinkage = shrinkage,
    interaction.depth = interaction.depth,
    n.minobsinnode = n.minobsinnode,
    cv.folds = 10
  )
  # compute RMSE
  sqrt(min(m$cv.error))
}

# perform search grid with functional programming
hyper_grid$rmse <- purrr::pmap_dbl(
  hyper_grid,
  ~ model_fit(
    n.trees = ..1,
    shrinkage = ..2,
    interaction.depth = ..3,
    n.minobsinnode = ..4
  )
)

# results
arrange(hyper_grid, rmse)
##   n.trees shrinkage interaction.depth n.minobsinnode  rmse
## 1    4000      0.05                 5              5 20699
## 2    4000      0.05                 3              5 20723
## 3    4000      0.05                 7              5 21021
## 4    4000      0.05                 3             10 21382
## 5    4000      0.05                 5             10 21915
## 6    4000      0.05                 5             15 21924
## 7    4000      0.05                 3             15 21943
## 8    4000      0.05                 7             10 21999
## 9    4000      0.05                 7             15 22348

### Stochastic GBM

# refined hyperparameter grid
hyper_grid <- list(
  sample_rate = c(0.5, 0.75, 1),              # row subsampling
  col_sample_rate = c(0.5, 0.75, 1),          # col subsampling for each split
  col_sample_rate_per_tree = c(0.5, 0.75, 1)  # col subsampling for each tree
)

# random grid search strategy
search_criteria <- list(
  strategy = "RandomDiscrete",
  stopping_metric = "mse",
  stopping_tolerance = 0.001,   
  stopping_rounds = 10,         
  max_runtime_secs = 60*60      
)

# perform grid search 
grid <- h2o.grid(
  algorithm = "gbm",
  grid_id = "gbm_grid",
  x = predictors, 
  y = response,
  training_frame = train_h2o,
  hyper_params = hyper_grid,
  ntrees = 6000,
  learn_rate = 0.01,
  max_depth = 7,
  min_rows = 5,
  nfolds = 10,
  stopping_rounds = 10,
  stopping_tolerance = 0,
  search_criteria = search_criteria,
  seed = 123
)

# collect the results and sort by our model performance metric of choice
grid_perf <- h2o.getGrid(
  grid_id = "gbm_grid", 
  sort_by = "mse", 
  decreasing = FALSE
)

grid_perf
## H2O Grid Details
## ================
## 
## Grid ID: gbm_grid 
## Used hyper parameters: 
##   -  col_sample_rate 
##   -  col_sample_rate_per_tree 
##   -  sample_rate 
## Number of models: 18 
## Number of failed models: 0 
## 
## Hyper-Parameter Search Summary: ordered by increasing mse
##    col_sample_rate col_sample_rate_per_tree sample_rate         model_ids                  mse
## 1              0.5                      0.5         0.5  gbm_grid_model_8  4.462965966345138E8
## 2              0.5                      1.0         0.5  gbm_grid_model_3  4.568248274796835E8
## 3              0.5                     0.75        0.75 gbm_grid_model_12 4.6466647244785947E8
## 4             0.75                      0.5        0.75  gbm_grid_model_5  4.689665768861389E8
## 5              1.0                     0.75         0.5 gbm_grid_model_14 4.7010349266737276E8
## 6              0.5                      0.5        0.75 gbm_grid_model_10  4.713882927949245E8
## 7             0.75                      1.0         0.5  gbm_grid_model_4  4.729884840420368E8
## 8              1.0                      1.0         0.5  gbm_grid_model_1  4.770705550988762E8
## 9              1.0                     0.75        0.75  gbm_grid_model_6 4.9292332262147874E8
## 10            0.75                      1.0        0.75 gbm_grid_model_13  4.985715082289563E8
## 11            0.75                      0.5         1.0  gbm_grid_model_2 5.0271257831462187E8
## 12            0.75                     0.75        0.75 gbm_grid_model_15 5.0981695262733763E8
## 13            0.75                     0.75         1.0  gbm_grid_model_9 5.3137490858680266E8
## 14            0.75                      1.0         1.0 gbm_grid_model_11   5.77518690995319E8
## 15             1.0                      1.0         1.0  gbm_grid_model_7  6.037512241688542E8
## 16             1.0                     0.75         1.0 gbm_grid_model_16 1.9742225720119803E9
## 17             0.5                      1.0        0.75 gbm_grid_model_17 4.1339991380839005E9
## 18             1.0                      0.5         1.0 gbm_grid_model_18  5.949489361558916E9

# Grab the model_id for the top model, chosen by cross validation error
best_model_id <- grid_perf@model_ids[[1]]
best_model <- h2o.getModel(best_model_id)

# Now let’s get performance metrics on the best model
h2o.performance(model = best_model, xval = TRUE)
## H2ORegressionMetrics: gbm
## ** Reported on cross-validation data. **
## ** 10-fold cross-validation on training data (Metrics computed for combined holdout predictions) **
## 
## MSE:  446296597
## RMSE:  21125.73
## MAE:  13045.95
## RMSLE:  0.1240542
## Mean Residual Deviance :  446296597

### XGBoost
library(recipes)
xgb_prep <- recipe(Sale_Price ~ ., data = ames_train) %>%
  step_integer(all_nominal()) %>%
  prep(training = ames_train, retain = TRUE) %>%
  juice()

X <- as.matrix(xgb_prep[setdiff(names(xgb_prep), "Sale_Price")])
Y <- xgb_prep$Sale_Price

set.seed(123)
ames_xgb <- xgb.cv(
  data = X,
  label = Y,
  nrounds = 6000,
  objective = "reg:linear",
  early_stopping_rounds = 50, 
  nfold = 10,
  params = list(
    eta = 0.1,
    max_depth = 3,
    min_child_weight = 3,
    subsample = 0.8,
    colsample_bytree = 1.0),
  verbose = 0
)  

# minimum test CV RMSE
min(ames_xgb$evaluation_log$test_rmse_mean)
## [1] 20488

# hyperparameter grid
hyper_grid <- expand.grid(
  eta = 0.01,
  max_depth = 3, 
  min_child_weight = 3,
  subsample = 0.5, 
  colsample_bytree = 0.5,
  gamma = c(0, 1, 10, 100, 1000),
  lambda = c(0, 1e-2, 0.1, 1, 100, 1000, 10000),
  alpha = c(0, 1e-2, 0.1, 1, 100, 1000, 10000),
  rmse = 0,          # a place to dump RMSE results
  trees = 0          # a place to dump required number of trees
)

# grid search
for(i in seq_len(nrow(hyper_grid))) {
  set.seed(123)
  m <- xgb.cv(
    data = X,
    label = Y,
    nrounds = 4000,
    objective = "reg:linear",
    early_stopping_rounds = 50, 
    nfold = 10,
    verbose = 0,
    params = list( 
      eta = hyper_grid$eta[i], 
      max_depth = hyper_grid$max_depth[i],
      min_child_weight = hyper_grid$min_child_weight[i],
      subsample = hyper_grid$subsample[i],
      colsample_bytree = hyper_grid$colsample_bytree[i],
      gamma = hyper_grid$gamma[i], 
      lambda = hyper_grid$lambda[i], 
      alpha = hyper_grid$alpha[i]
    ) 
  )
  hyper_grid$rmse[i] <- min(m$evaluation_log$test_rmse_mean)
  hyper_grid$trees[i] <- m$best_iteration
}

# results
hyper_grid %>%
  filter(rmse > 0) %>%
  arrange(rmse) %>%
  glimpse()
## Observations: 98
## Variables: 10
## $ eta              <dbl> 0.01, 0.01, 0.01, 0.01, 0.01, 0.0…
## $ max_depth        <dbl> 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, …
## $ min_child_weight <dbl> 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, …
## $ subsample        <dbl> 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5…
## $ colsample_bytree <dbl> 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5…
## $ gamma            <dbl> 0, 1, 10, 100, 1000, 0, 1, 10, 10…
## $ lambda           <dbl> 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, …
## $ alpha            <dbl> 0.00, 0.00, 0.00, 0.00, 0.00, 0.1…
## $ rmse             <dbl> 20488, 20488, 20488, 20488, 20488…
## $ trees            <dbl> 3944, 3944, 3944, 3944, 3944, 381…

# optimal parameter list
params <- list(
  eta = 0.01,
  max_depth = 3,
  min_child_weight = 3,
  subsample = 0.5,
  colsample_bytree = 0.5
)

# train final model
xgb.fit.final <- xgboost(
  params = params,
  data = X,
  label = Y,
  nrounds = 3944,
  objective = "reg:linear",
  verbose = 0
)

### Feature interpretation
# variable importance plot
vip::vip(xgb.fit.final) 
