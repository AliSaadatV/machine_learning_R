# Helper packages
library(dplyr)    # for data manipulation
library(ggplot2)  # for awesome graphics
library(visdat)   # for additional visualizations

# Feature engineering packages
library(caret)    # for various ML tasks
library(recipes)  # for feature engineering tasks

# Stratified sampling with the rsample package
set.seed(123)
ames <- AmesHousing::make_ames()
split <- initial_split(ames, prop = 0.7, 
                       strata = "Sale_Price")
ames_train  <- training(split)
ames_test   <- testing(split)

# Thinking of feature engineering as a blueprint forces us to think of the ordering of our preprocessing steps.
# Although each particular problem requires you to think of the effects of sequential preprocessing, there are some general suggestions that you should consider:
#   
# 1) If using a log or Box-Cox transformation, don’t center the data first or do any operations that might make the data non-positive. Alternatively, use the Yeo-Johnson transformation so you don’t have to worry about this.
# 2) One-hot or dummy encoding typically results in sparse data which many algorithms can operate efficiently on. If you standardize sparse data you will create dense data and you loose the computational efficiency. Consequently, it’s often preferred to standardize your numeric features and then one-hot/dummy encode.
# 3) If you are lumping infrequently occurring categories together, do so before one-hot/dummy encoding.
# 4) Although you can perform dimension reduction procedures on categorical features, it is common to primarily do so on numeric features when doing so for feature engineering purposes.
# 
# While your project’s needs may vary, here is a suggested order of potential steps that should work for most problems:
#   
# 1) Filter out zero or near-zero variance features.
# 2) Perform imputation if required.
# 3) Normalize to resolve numeric feature skewness.
# 4) Standardize (center and scale) numeric features.
# 5) Perform dimension reduction (e.g., PCA) on numeric features.
# 6) One-hot or dummy encode categorical features.

# For the 'ames' data, we perform:
# 1) Remove near-zero variance features that are categorical (aka nominal).
# 2) Ordinal encode our quality-based features (which are inherently ordinal).
# 3) Center and scale (i.e., standardize) all numeric features.
# 4) Perform dimension reduction by applying PCA to all numeric features.

blueprint <- recipe(Sale_Price ~ ., data = ames_train) %>%
  step_nzv(all_nominal()) %>%
  step_integer(matches("Qual|Cond|QC|Qu")) %>%
  step_center(all_numeric(), -all_outcomes()) %>%
  step_scale(all_numeric(), -all_outcomes()) %>%
  step_dummy(all_nominal(), -all_outcomes(), one_hot = TRUE)

# Specify resampling plan
cv <- trainControl(
  method = "repeatedcv", 
  number = 10, 
  repeats = 5
)

# Construct grid of hyperparameter values
hyper_grid <- expand.grid(k = seq(2, 25, by = 1))

# Tune a knn model using grid search
knn_fit2 <- train(
  blueprint, 
  data = ames_train, 
  method = "knn", 
  trControl = cv, 
  tuneGrid = hyper_grid,
  metric = "RMSE"
)
