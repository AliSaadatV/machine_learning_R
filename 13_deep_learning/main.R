# Helper packages
library(dplyr)         # for basic data wrangling

# Modeling packages
library(keras)         # for fitting DNNs
library(tfruns)        # for additional grid search & model training functions

# Modeling helper package - not necessary for reproducibility
library(tfestimators)  # provides grid search & model training interface

# Import MNIST training data
mnist <- dslabs::read_mnist()
mnist_x <- mnist$train$images
mnist_y <- mnist$train$labels

# Rename columns and standardize feature values
colnames(mnist_x) <- paste0("V", 1:ncol(mnist_x))
mnist_x <- mnist_x / 255

# One-hot encode response
mnist_y <- to_categorical(mnist_y, 10)

### basic implementation
model <- keras_model_sequential() %>%
  layer_dense(units = 128, input_shape = ncol(mnist_x)) %>%
  layer_dense(units = 64) %>%
  layer_dense(units = 10)

### implementation with activation functions
model <- keras_model_sequential() %>%
  layer_dense(units = 128, activation = "relu", input_shape = p) %>%
  layer_dense(units = 64, activation = "relu") %>%
  layer_dense(units = 10, activation = "softmax")

### model training
model <- keras_model_sequential() %>%
  
  # Network architecture
  layer_dense(units = 128, activation = "relu", input_shape = ncol(mnist_x)) %>%
  layer_dense(units = 64, activation = "relu") %>%
  layer_dense(units = 10, activation = "softmax") %>%
  
  # Backpropagation
  compile(
    loss = 'categorical_crossentropy',
    optimizer = optimizer_rmsprop(),
    metrics = c('accuracy')
  )

# Train the model
fit1 <- model %>%
  fit(
    x = mnist_x,
    y = mnist_y,
    epochs = 25,
    batch_size = 128,
    validation_split = 0.2,
    verbose = FALSE
  )

# Display output
fit1
## Trained on 48,000 samples, validated on 12,000 samples (batch_size=128, epochs=25)
## Final epoch (plot to see history):
## val_loss: 0.1512
##  val_acc: 0.9773
##     loss: 0.002308
##      acc: 0.9994
plot(fit1)

### batch normalization
model_w_norm <- keras_model_sequential() %>%
  
  # Network architecture with batch normalization
  layer_dense(units = 256, activation = "relu", input_shape = ncol(mnist_x)) %>%
  layer_batch_normalization() %>%
  layer_dense(units = 128, activation = "relu") %>%
  layer_batch_normalization() %>%
  layer_dense(units = 64, activation = "relu") %>%
  layer_batch_normalization() %>%
  layer_dense(units = 10, activation = "softmax") %>%
  
  # Backpropagation
  compile(
    loss = "categorical_crossentropy",
    optimizer = optimizer_rmsprop(),
    metrics = c("accuracy")
  )

### regularization with L1 or L2
model_w_reg <- keras_model_sequential() %>%
  
  # Network architecture with L1 regularization and batch normalization
  layer_dense(units = 256, activation = "relu", input_shape = ncol(mnist_x),
              kernel_regularizer = regularizer_l2(0.001)) %>%
  layer_batch_normalization() %>%
  layer_dense(units = 128, activation = "relu", 
              kernel_regularizer = regularizer_l2(0.001)) %>%
  layer_batch_normalization() %>%
  layer_dense(units = 64, activation = "relu", 
              kernel_regularizer = regularizer_l2(0.001)) %>%
  layer_batch_normalization() %>%
  layer_dense(units = 10, activation = "softmax") %>%
  
  # Backpropagation
  compile(
    loss = "categorical_crossentropy",
    optimizer = optimizer_rmsprop(),
    metrics = c("accuracy")
  )

### regularization with dropout
model_w_drop <- keras_model_sequential() %>%
  
  # Network architecture with 20% dropout
  layer_dense(units = 256, activation = "relu", input_shape = ncol(mnist_x)) %>%
  layer_dropout(rate = 0.2) %>%
  layer_dense(units = 128, activation = "relu") %>%
  layer_dropout(rate = 0.2) %>%
  layer_dense(units = 64, activation = "relu") %>%
  layer_dropout(rate = 0.2) %>%
  layer_dense(units = 10, activation = "softmax") %>%
  
  # Backpropagation
  compile(
    loss = "categorical_crossentropy",
    optimizer = optimizer_rmsprop(),
    metrics = c("accuracy")
  )

### adjusting learning rate
model_w_adj_lrn <- keras_model_sequential() %>%
  layer_dense(units = 256, activation = "relu", input_shape = ncol(mnist_x)) %>%
  layer_batch_normalization() %>%
  layer_dropout(rate = 0.4) %>%
  layer_dense(units = 128, activation = "relu") %>%
  layer_batch_normalization() %>%
  layer_dropout(rate = 0.3) %>%
  layer_dense(units = 64, activation = "relu") %>%
  layer_batch_normalization() %>%
  layer_dropout(rate = 0.2) %>%
  layer_dense(units = 10, activation = "softmax") %>%
  compile(
    loss = 'categorical_crossentropy',
    optimizer = optimizer_adam(),
    metrics = c('accuracy')
  ) %>%
  fit(
    x = mnist_x,
    y = mnist_y,
    epochs = 35,
    batch_size = 128,
    validation_split = 0.2,
    callbacks = list(
      callback_early_stopping(patience = 5),
      callback_reduce_lr_on_plateau(factor = 0.05)
    ),
    verbose = FALSE
  )

model_w_adj_lrn
## Trained on 48,000 samples, validated on 12,000 samples (batch_size=128, epochs=20)
## Final epoch (plot to see history):
## val_loss: 0.07223
##  val_acc: 0.9808
##     loss: 0.05366
##      acc: 0.9832
##       lr: 0.001

# Optimal
min(model_w_adj_lrn$metrics$val_loss)
## [1] 0.0699492
max(model_w_adj_lrn$metrics$val_acc)
## [1] 0.981

# Learning rate
plot(model_w_adj_lrn)

### model tunning with random grid search 
### take a look at the mnist-grid-search.R

runs <- tuning_run("mnist-grid-search.R", 
                   flags = list(
                     nodes1 = c(64, 128, 256),
                     nodes2 = c(64, 128, 256),
                     nodes3 = c(64, 128, 256),
                     dropout1 = c(0.2, 0.3, 0.4),
                     dropout2 = c(0.2, 0.3, 0.4),
                     dropout3 = c(0.2, 0.3, 0.4),
                     optimizer = c("rmsprop", "adam"),
                     lr_annealing = c(0.1, 0.05)
                   ),
                   sample = 0.05
)

runs %>% 
  filter(metric_val_loss == min(metric_val_loss)) %>% 
  glimpse()
## Observations: 1
## Variables: 31
## $ run_dir            <chr> "runs/2019-04-27T14-44-38Z"
## $ metric_loss        <dbl> 0.0598
## $ metric_acc         <dbl> 0.9806
## $ metric_val_loss    <dbl> 0.0686
## $ metric_val_acc     <dbl> 0.9806
## $ flag_nodes1        <int> 256
## $ flag_nodes2        <int> 128
## $ flag_nodes3        <int> 256
## $ flag_dropout1      <dbl> 0.4
## $ flag_dropout2      <dbl> 0.2
## $ flag_dropout3      <dbl> 0.3
## $ flag_optimizer     <chr> "adam"
## $ flag_lr_annealing  <dbl> 0.05
## $ samples            <int> 48000
## $ validation_samples <int> 12000
## $ batch_size         <int> 128
## $ epochs             <int> 35
## $ epochs_completed   <int> 17
## $ metrics            <chr> "runs/2019-04-27T14-44-38Z/tfruns.d/metrics.json"
## $ model              <chr> "Model\n_______________________________________________________…
## $ loss_function      <chr> "categorical_crossentropy"
## $ optimizer          <chr> "<tensorflow.python.keras.optimizers.Adam>"
## $ learning_rate      <dbl> 0.001
## $ script             <chr> "mnist-grid-search.R"
## $ start              <dttm> 2019-04-27 14:44:38
## $ end                <dttm> 2019-04-27 14:45:39
## $ completed          <lgl> TRUE
## $ output             <chr> "\n> #' Trains a feedforward DL model on the MNIST dataset.\n> …
## $ source_code        <chr> "runs/2019-04-27T14-44-38Z/tfruns.d/source.tar.gz"
## $ context            <chr> "local"
## $ type               <chr> "training"