# Helper packages
library(dplyr)    # for data manipulation
library(ggplot2)  # for data visualization

# Modeling packages
library(h2o)  # for fitting autoencoders

mnist <- dslabs::read_mnist()
names(mnist)
## [1] "train" "test"

h2o.no_progress()  # turn off progress bars
h2o.init(max_mem_size = "5g")  # initialize H2O instance

### basic autoencoder
# Convert mnist features to an h2o input data set
features <- as.h2o(mnist$train$images)

# Train an autoencoder
ae1 <- h2o.deeplearning(
  x = seq_along(features),
  training_frame = features,
  autoencoder = TRUE,
  hidden = 2,
  activation = 'Tanh',
  sparse = TRUE
)

# Extract the deep features
ae1_codings <- h2o.deepfeatures(ae1, features, layer = 1)
ae1_codings
##     DF.L1.C1    DF.L1.C2
## 1 -0.1558956 -0.06456967
## 2  0.3778544 -0.61518649
## 3  0.2002303  0.31214266
## 4 -0.6955515  0.13225607
## 5  0.1912538  0.59865392
## 6  0.2310982  0.20322605
## 
## [60000 rows x 2 columns]

### stacked autoencoders
# Hyperparameter search grid
hyper_grid <- list(hidden = list(
  c(50),
  c(100), 
  c(300, 100, 300),
  c(100, 50, 100),
  c(250, 100, 50, 100, 250)
))

# Execute grid search
ae_grid <- h2o.grid(
  algorithm = 'deeplearning',
  x = seq_along(features),
  training_frame = features,
  grid_id = 'autoencoder_grid',
  autoencoder = TRUE,
  activation = 'Tanh',
  hyper_params = hyper_grid,
  sparse = TRUE,
  ignore_const_cols = FALSE,
  seed = 123
)

# Print grid details
h2o.getGrid('autoencoder_grid', sort_by = 'mse', decreasing = FALSE)
## H2O Grid Details
## ================
## 
## Grid ID: autoencoder_grid 
## Used hyper parameters: 
##   -  hidden 
## Number of models: 5 
## Number of failed models: 0 
## 
## Hyper-Parameter Search Summary: ordered by increasing mse
##                     hidden                 model_ids                  mse
## 1                    [100] autoencoder_grid3_model_2  0.00674637890553651
## 2          [300, 100, 300] autoencoder_grid3_model_3  0.00830502966843272
## 3           [100, 50, 100] autoencoder_grid3_model_4 0.011215307972822733
## 4                     [50] autoencoder_grid3_model_1 0.012450109189122541
## 5 [250, 100, 50, 100, 250] autoencoder_grid3_model_5 0.014410280145600972

### Visualizing the reconstructed images
# Get sampled test images
index <- sample(1:nrow(mnist$test$images), 4)
sampled_digits <- mnist$test$images[index, ]
colnames(sampled_digits) <- paste0("V", seq_len(ncol(sampled_digits)))

# Predict reconstructed pixel values
best_model_id <- ae_grid@model_ids[[1]]
best_model <- h2o.getModel(best_model_id)
reconstructed_digits <- predict(best_model, as.h2o(sampled_digits))
names(reconstructed_digits) <- paste0("V", seq_len(ncol(reconstructed_digits)))

combine <- rbind(sampled_digits, as.matrix(reconstructed_digits))

# Plot original versus reconstructed
par(mfrow = c(1, 3), mar=c(1, 1, 1, 1))
layout(matrix(seq_len(nrow(combine)), 4, 2, byrow = FALSE))
for(i in seq_len(nrow(combine))) {
  image(matrix(combine[i, ], 28, 28)[, 28:1], xaxt="n", yaxt="n")
}

### Sparse autoencoders

ae100_codings <- h2o.deepfeatures(best_model, features, layer = 1)
ae100_codings %>% 
  as.data.frame() %>% 
  tidyr::gather() %>%
  summarize(average_activation = mean(value))
##   average_activation
## 1        -0.00677801

# Hyperparameter search grid
hyper_grid <- list(sparsity_beta = c(0.01, 0.05, 0.1, 0.2))

# Execute grid search
ae_sparsity_grid <- h2o.grid(
  algorithm = 'deeplearning',
  x = seq_along(features),
  training_frame = features,
  grid_id = 'sparsity_grid',
  autoencoder = TRUE,
  hidden = 100,
  activation = 'Tanh',
  hyper_params = hyper_grid,
  sparse = TRUE,
  average_activation = -0.1,
  ignore_const_cols = FALSE,
  seed = 123
)

# Print grid details
h2o.getGrid('sparsity_grid', sort_by = 'mse', decreasing = FALSE)
## H2O Grid Details
## ================
## 
## Grid ID: sparsity_grid 
## Used hyper parameters: 
##   -  sparsity_beta 
## Number of models: 4 
## Number of failed models: 0 
## 
## Hyper-Parameter Search Summary: ordered by increasing mse
##   sparsity_beta             model_ids                  mse
## 1          0.01 sparsity_grid_model_1 0.012982916169006953
## 2           0.2 sparsity_grid_model_4  0.01321464889160263
## 3          0.05 sparsity_grid_model_2  0.01337749148043942
## 4           0.1 sparsity_grid_model_3 0.013516631653257992

### denoising autoencoders
# Train a denoise autoencoder
denoise_ae <- h2o.deeplearning(
  x = seq_along(features),
  training_frame = inputs_currupted_gaussian,
  validation_frame = features,
  autoencoder = TRUE,
  hidden = 100,
  activation = 'Tanh',
  sparse = TRUE
)

# Print performance
h2o.performance(denoise_ae, valid = TRUE)
## H2OAutoEncoderMetrics: deeplearning
## ** Reported on validation data. **
## 
## Validation Set Metrics: 
## =====================
## 
## MSE: (Extract with `h2o.mse`) 0.02048465
## RMSE: (Extract with `h2o.rmse`) 0.1431246

### anamoly detection
# Extract reconstruction errors
(reconstruction_errors <- h2o.anomaly(best_model, features))
##   Reconstruction.MSE
## 1        0.009879666
## 2        0.006485201
## 3        0.017470110
## 4        0.002339352
## 5        0.006077669
## 6        0.007171287
## 
## [60000 rows x 1 column]

# Plot distribution
reconstruction_errors <- as.data.frame(reconstruction_errors)
ggplot(reconstruction_errors, aes(Reconstruction.MSE)) +
  geom_histogram()


