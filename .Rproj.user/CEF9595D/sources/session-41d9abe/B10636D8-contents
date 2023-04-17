# Helper packages
library(rsample)   # for creating our train-test splits
library(recipes)   # for minor feature engineering tasks

# Modeling packages
library(h2o)       # for fitting stacked models

# Load and split the Ames housing data
ames <- AmesHousing::make_ames()
set.seed(123)  # for reproducibility
split <- initial_split(ames, strata = "Sale_Price")
ames_train <- training(split)
ames_test <- testing(split)

# Make sure we have consistent categorical levels
blueprint <- recipe(Sale_Price ~ ., data = ames_train) %>%
  step_other(all_nominal(), threshold = 0.005)

# Create training & test sets for h2o
train_h2o <- prep(blueprint, training = ames_train, retain = TRUE) %>%
  juice() %>%
  as.h2o()
test_h2o <- prep(blueprint, training = ames_train) %>%
  bake(new_data = ames_test) %>%
  as.h2o()

# Get response and feature names
Y <- "Sale_Price"
X <- setdiff(names(ames_train), Y)

### Stack existing models
# Train & cross-validate a GLM model
best_glm <- h2o.glm(
  x = X, y = Y, training_frame = train_h2o, alpha = 0.1,
  remove_collinear_columns = TRUE, nfolds = 10, fold_assignment = "Modulo",
  keep_cross_validation_predictions = TRUE, seed = 123
)

# Train & cross-validate a RF model
best_rf <- h2o.randomForest(
  x = X, y = Y, training_frame = train_h2o, ntrees = 1000, mtries = 20,
  max_depth = 30, min_rows = 1, sample_rate = 0.8, nfolds = 10,
  fold_assignment = "Modulo", keep_cross_validation_predictions = TRUE,
  seed = 123, stopping_rounds = 50, stopping_metric = "RMSE",
  stopping_tolerance = 0
)

# Train & cross-validate a GBM model
best_gbm <- h2o.gbm(
  x = X, y = Y, training_frame = train_h2o, ntrees = 5000, learn_rate = 0.01,
  max_depth = 7, min_rows = 5, sample_rate = 0.8, nfolds = 10,
  fold_assignment = "Modulo", keep_cross_validation_predictions = TRUE,
  seed = 123, stopping_rounds = 50, stopping_metric = "RMSE",
  stopping_tolerance = 0
)

# Train & cross-validate an XGBoost model
best_xgb <- h2o.xgboost(
  x = X, y = Y, training_frame = train_h2o, ntrees = 5000, learn_rate = 0.05,
  max_depth = 3, min_rows = 3, sample_rate = 0.8, categorical_encoding = "Enum",
  nfolds = 10, fold_assignment = "Modulo", 
  keep_cross_validation_predictions = TRUE, seed = 123, stopping_rounds = 50,
  stopping_metric = "RMSE", stopping_tolerance = 0
)

# Train a stacked tree ensemble
ensemble_tree <- h2o.stackedEnsemble(
  x = X, y = Y, training_frame = train_h2o, model_id = "my_tree_ensemble",
  base_models = list(best_glm, best_rf, best_gbm, best_xgb),
  metalearner_algorithm = "drf"
)

# Get results from base learners
get_rmse <- function(model) {
  results <- h2o.performance(model, newdata = test_h2o)
  results@metrics$RMSE
}
list(best_glm, best_rf, best_gbm, best_xgb) %>%
  purrr::map_dbl(get_rmse)
## [1] 30024.67 23075.24 20859.92 21391.20

# Stacked results
h2o.performance(ensemble_tree, newdata = test_h2o)@metrics$RMSE
## [1] 20664.56

# correlation of base learners
data.frame(
  GLM_pred = as.vector(h2o.getFrame(best_glm@model$cross_validation_holdout_predictions_frame_id$name)),
  RF_pred = as.vector(h2o.getFrame(best_rf@model$cross_validation_holdout_predictions_frame_id$name)),
  GBM_pred = as.vector(h2o.getFrame(best_gbm@model$cross_validation_holdout_predictions_frame_id$name)),
  XGB_pred = as.vector(h2o.getFrame(best_xgb@model$cross_validation_holdout_predictions_frame_id$name))
) %>% cor()
##           GLM_pred   RF_pred  GBM_pred  XGB_pred
## GLM_pred 1.0000000 0.9390229 0.9291982 0.9345048
## RF_pred  0.9390229 1.0000000 0.9920349 0.9821944
## GBM_pred 0.9291982 0.9920349 1.0000000 0.9854160
## XGB_pred 0.9345048 0.9821944 0.9854160 1.0000000

### stacking a grid seach
# Define GBM hyperparameter grid
hyper_grid <- list(
  max_depth = c(1, 3, 5),
  min_rows = c(1, 5, 10),
  learn_rate = c(0.01, 0.05, 0.1),
  learn_rate_annealing = c(0.99, 1),
  sample_rate = c(0.5, 0.75, 1),
  col_sample_rate = c(0.8, 0.9, 1)
)

# Define random grid search criteria
search_criteria <- list(
  strategy = "RandomDiscrete",
  max_models = 25
)

# Build random grid search 
random_grid <- h2o.grid(
  algorithm = "gbm", grid_id = "gbm_grid", x = X, y = Y,
  training_frame = train_h2o, hyper_params = hyper_grid,
  search_criteria = search_criteria, ntrees = 5000, stopping_metric = "RMSE",     
  stopping_rounds = 10, stopping_tolerance = 0, nfolds = 10, 
  fold_assignment = "Modulo", keep_cross_validation_predictions = TRUE,
  seed = 123
)
If we look at the grid search models we see that the cross-validated RMSE ranges from 20756–57826

# Sort results by RMSE
h2o.getGrid(
  grid_id = "gbm_grid", 
  sort_by = "rmse"
)
## H2O Grid Details
## ================
## 
## Grid ID: gbm_grid 
## Used hyper parameters: 
##   -  col_sample_rate 
##   -  learn_rate 
##   -  learn_rate_annealing 
##   -  max_depth 
##   -  min_rows 
##   -  sample_rate 
## Number of models: 25 
## Number of failed models: 0 
## 
## Hyper-Parameter Search Summary: ordered by increasing rmse
##   col_sample_rate learn_rate learn_rate_annealing max_depth min_rows sample_rate         model_ids               rmse
## 1             0.9       0.01                  1.0         3      1.0         1.0 gbm_grid_model_20  20756.16775065606
## 2             0.9       0.01                  1.0         5      1.0        0.75  gbm_grid_model_2 21188.696088824694
## 3             0.9        0.1                  1.0         3      1.0        0.75  gbm_grid_model_5 21203.753908665003
## 4             0.8       0.01                  1.0         5      5.0         1.0 gbm_grid_model_16 21704.257699437963
## 5             1.0        0.1                 0.99         3      1.0         1.0 gbm_grid_model_17 21710.275753497197
## 
## ---
##    col_sample_rate learn_rate learn_rate_annealing max_depth min_rows sample_rate         model_ids               rmse
## 20             1.0       0.01                  1.0         1     10.0        0.75 gbm_grid_model_11 26164.879525289896
## 21             0.8       0.01                 0.99         3      1.0        0.75 gbm_grid_model_15  44805.63843296435
## 22             1.0       0.01                 0.99         3     10.0         1.0 gbm_grid_model_18 44854.611500840605
## 23             0.8       0.01                 0.99         1     10.0         1.0 gbm_grid_model_21 57797.874642563846
## 24             0.9       0.01                 0.99         1     10.0        0.75 gbm_grid_model_10  57809.60302408739
## 25             0.8       0.01                 0.99         1      5.0        0.75  gbm_grid_model_4  57826.30370545089

# Grab the model_id for the top model, chosen by validation error
best_model_id <- random_grid_perf@model_ids[[1]]
best_model <- h2o.getModel(best_model_id)
h2o.performance(best_model, newdata = test_h2o)
## H2ORegressionMetrics: gbm
## 
## MSE:  466551295
## RMSE:  21599.8
## MAE:  13697.78
## RMSLE:  0.1090604
## Mean Residual Deviance :  466551295

# Train a stacked ensemble using the GBM grid
ensemble <- h2o.stackedEnsemble(
  x = X, y = Y, training_frame = train_h2o, model_id = "ensemble_gbm_grid",
  base_models = random_grid@model_ids, metalearner_algorithm = "gbm"
)

# Eval ensemble performance on a test set
h2o.performance(ensemble, newdata = test_h2o)
## H2ORegressionMetrics: stackedensemble
## 
## MSE:  469579433
## RMSE:  21669.78
## MAE:  13499.93
## RMSLE:  0.1061244
## Mean Residual Deviance :  469579433

### Automated machine learning
# Use AutoML to find a list of candidate models (i.e., leaderboard)
auto_ml <- h2o.automl(
  x = X, y = Y, training_frame = train_h2o, nfolds = 5, 
  max_runtime_secs = 60 * 120, max_models = 50,
  keep_cross_validation_predictions = TRUE, sort_metric = "RMSE", seed = 123,
  stopping_rounds = 50, stopping_metric = "RMSE", stopping_tolerance = 0
)

# Assess the leader board; the following truncates the results to show the top 
# and bottom 15 models. You can get the top model with auto_ml@leader
auto_ml@leaderboard %>% 
  as.data.frame() %>%
  dplyr::select(model_id, rmse) %>%
  dplyr::slice(1:25)
##                                               model_id   rmse
## 1                     XGBoost_1_AutoML_20190220_084553   22229.97
## 2            GBM_grid_1_AutoML_20190220_084553_model_1   22437.26
## 3            GBM_grid_1_AutoML_20190220_084553_model_3   22777.57
## 4                         GBM_2_AutoML_20190220_084553   22785.60
## 5                         GBM_3_AutoML_20190220_084553   23133.59
## 6                         GBM_4_AutoML_20190220_084553   23185.45
## 7                     XGBoost_2_AutoML_20190220_084553   23199.68
## 8                     XGBoost_1_AutoML_20190220_075753   23231.28
## 9                         GBM_1_AutoML_20190220_084553   23326.57
## 10           GBM_grid_1_AutoML_20190220_075753_model_2   23330.42
## 11                    XGBoost_3_AutoML_20190220_084553   23475.23
## 12       XGBoost_grid_1_AutoML_20190220_084553_model_3   23550.04
## 13      XGBoost_grid_1_AutoML_20190220_075753_model_15   23640.95
## 14       XGBoost_grid_1_AutoML_20190220_084553_model_8   23646.66
## 15       XGBoost_grid_1_AutoML_20190220_084553_model_6   23682.37
## ...                                                ...        ...
## 65           GBM_grid_1_AutoML_20190220_084553_model_5   33971.32
## 66           GBM_grid_1_AutoML_20190220_075753_model_8   34489.39
## 67  DeepLearning_grid_1_AutoML_20190220_084553_model_3   36591.73
## 68           GBM_grid_1_AutoML_20190220_075753_model_6   36667.56
## 69      XGBoost_grid_1_AutoML_20190220_084553_model_13   40416.32
## 70           GBM_grid_1_AutoML_20190220_075753_model_9   47744.43
## 71    StackedEnsemble_AllModels_AutoML_20190220_084553   49856.66
## 72    StackedEnsemble_AllModels_AutoML_20190220_075753   59127.09
## 73 StackedEnsemble_BestOfFamily_AutoML_20190220_084553   76714.90
## 74 StackedEnsemble_BestOfFamily_AutoML_20190220_075753   76748.40
## 75           GBM_grid_1_AutoML_20190220_075753_model_5   78465.26
## 76           GBM_grid_1_AutoML_20190220_075753_model_3   78535.34
## 77           GLM_grid_1_AutoML_20190220_075753_model_1   80284.34
## 78           GLM_grid_1_AutoML_20190220_084553_model_1   80284.34
## 79       XGBoost_grid_1_AutoML_20190220_075753_model_4   92559.44
## 80      XGBoost_grid_1_AutoML_20190220_075753_model_10  125384.88
