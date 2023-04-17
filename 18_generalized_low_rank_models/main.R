# Helper packages
library(dplyr)    # for data manipulation
library(ggplot2)  # for data visualization
library(tidyr)    # for data reshaping

# Modeling packages
library(h2o)  # for fitting GLRMs

url <- "https://koalaverse.github.io/homlr/data/my_basket.csv"
my_basket <- readr::read_csv(url)

h2o.no_progress()  # turn off progress bars
h2o.init(max_mem_size = "5g")  # connect to H2O instance

# convert data to h2o object
my_basket.h2o <- as.h2o(my_basket)

# run basic GLRM
basic_glrm <- h2o.glrm(
  training_frame = my_basket.h2o,
  k = 20, 
  loss = "Quadratic",
  regularization_x = "None", 
  regularization_y = "None", 
  transform = "STANDARDIZE", 
  max_iterations = 2000,
  seed = 123
)

# get top level summary information on our model
summary(basic_glrm)
## Model Details:
## ==============
## 
## H2ODimReductionModel: glrm
## Model Key:  GLRM_model_R_1538746363268_1 
## Model Summary: 
##   number_of_iterations final_step_size final_objective_value
## 1                  901         0.36373           31004.59190
## 
## H2ODimReductionMetrics: glrm
## ** Reported on training data. **
## 
## Sum of Squared Error (Numeric):  31004.59
## Misclassification Error (Categorical):  0
## Number of Numeric Entries:  84000
## Number of Categorical Entries:  0
## 
## 
## 
## Scoring History: 
##             timestamp   duration iterations step_size   objective
## 1 2018-10-05 09:32:54  1.106 sec          0   0.66667 67533.03413
## 2 2018-10-05 09:32:54  1.149 sec          1   0.70000 49462.95972
## 3 2018-10-05 09:32:55  1.226 sec          2   0.46667 49462.95972
## 4 2018-10-05 09:32:55  1.257 sec          3   0.31111 49462.95972
## 5 2018-10-05 09:32:55  1.289 sec          4   0.32667 41215.38164
## 
## ---
##               timestamp   duration iterations step_size   objective
## 896 2018-10-05 09:33:22 28.535 sec        895   0.28499 31004.59207
## 897 2018-10-05 09:33:22 28.566 sec        896   0.29924 31004.59202
## 898 2018-10-05 09:33:22 28.597 sec        897   0.31421 31004.59197
## 899 2018-10-05 09:33:22 28.626 sec        898   0.32992 31004.59193
## 900 2018-10-05 09:33:22 28.655 sec        899   0.34641 31004.59190
## 901 2018-10-05 09:33:22 28.685 sec        900   0.36373 31004.59190

# Create plot to see if results converged - if it did not converge, 
# consider increasing iterations or using different algorithm
plot(basic_glrm)

# amount of variance explained by each archetype (aka "pc")
basic_glrm@model$importance
## Importance of components: 
##                             pc1      pc2      pc3      pc4      pc5      pc6      pc7
## Standard deviation     1.513919 1.473768 1.459114 1.440635 1.435279 1.411544 1.253307
## Proportion of Variance 0.054570 0.051714 0.050691 0.049415 0.049048 0.047439 0.037400
## Cumulative Proportion  0.054570 0.106284 0.156975 0.206390 0.255438 0.302878 0.340277
##                             pc8      pc9     pc10     pc11     pc12     pc13     pc14
## Standard deviation     1.026387 1.010238 1.007253 0.988724 0.985320 0.970453 0.964303
## Proportion of Variance 0.025083 0.024300 0.024156 0.023276 0.023116 0.022423 0.022140
## Cumulative Proportion  0.365360 0.389659 0.413816 0.437091 0.460207 0.482630 0.504770
##                            pc15     pc16     pc17     pc18     pc19     pc20
## Standard deviation     0.951610 0.947978 0.944826 0.932943 0.931745 0.924206
## Proportion of Variance 0.021561 0.021397 0.021255 0.020723 0.020670 0.020337
## Cumulative Proportion  0.526331 0.547728 0.568982 0.589706 0.610376 0.630713


data.frame(
  PC  = basic_glrm@model$importance %>% seq_along(),
  PVE = basic_glrm@model$importance %>% .[2,] %>% unlist(),
  CVE = basic_glrm@model$importance %>% .[3,] %>% unlist()
) %>%
  gather(metric, variance_explained, -PC) %>%
  ggplot(aes(PC, variance_explained)) +
  geom_point() +
  facet_wrap(~ metric, ncol = 1, scales = "free")

# influence of each feature on archetypes
t(basic_glrm@model$archetypes)[1:5, 1:5]
##              Arch1      Arch2      Arch3      Arch4       Arch5
## 7up     -0.5783538 -1.5705325  0.9906612 -0.9306704  0.17552643
## lasagna  0.2196728  0.1213954 -0.7068851  0.8436524  3.56206178
## pepsi   -0.2504310 -0.8156136 -0.7669562 -1.2551630 -0.47632696
## yop     -0.1856632  0.4000083 -0.4855958  1.1598919 -0.26142763
## redwine -0.1372589 -0.1059148 -0.9579530  0.4641668 -0.08539977

p1 <- t(basic_glrm@model$archetypes) %>% 
  as.data.frame() %>% 
  mutate(feature = row.names(.)) %>%
  ggplot(aes(Arch1, reorder(feature, Arch1))) +
  geom_point()

p2 <- t(basic_glrm@model$archetypes) %>% 
  as.data.frame() %>% 
  mutate(feature = row.names(.)) %>%
  ggplot(aes(Arch1, Arch2, label = feature)) +
  geom_text()

gridExtra::grid.arrange(p1, p2, nrow = 1)

# Re-run model with k = 8
k8_glrm <- h2o.glrm(
  training_frame = my_basket.h2o,
  k = 8, 
  loss = "Quadratic",
  regularization_x = "None", 
  regularization_y = "None", 
  transform = "STANDARDIZE", 
  max_iterations = 2000,
  seed = 123
)

# Reconstruct to see how well the model did
my_reconstruction <- h2o.reconstruct(k8_glrm, my_basket.h2o, reverse_transform = TRUE)

# Raw predicted values
my_reconstruction[1:5, 1:5]
##   reconstr_7up reconstr_lasagna reconstr_pepsi reconstr_yop reconstr_red-wine
## 1  0.025595726      -0.06657864    -0.03813350 -0.012225807        0.03814142
## 2 -0.041778553       0.02401056    -0.05225379 -0.052248809       -0.05487031
## 3  0.012373600       0.04849545     0.05760424 -0.009878976        0.02492625
## 4  0.338875544       0.00577020     0.48763580  0.187669229        0.53358405
## 5  0.003869531       0.05394523     0.07655745 -0.010977765        0.51779314
## 
## [5 rows x 5 columns]

# Round values to whole integers
my_reconstruction[1:5, 1:5] %>% round(0)
##   reconstr_7up reconstr_lasagna reconstr_pepsi reconstr_yop reconstr_red-wine
## 1            0                0              0            0                 0
## 2            0                0              0            0                 0
## 3            0                0              0            0                 0
## 4            0                0              0            0                 1
## 5            0                0              0            0                 1
## 
## [5 rows x 5 columns]

### tunning
# Use non-negative regularization
k8_glrm_regularized <- h2o.glrm(
  training_frame = my_basket.h2o,
  k = 8, 
  loss = "Quadratic",
  regularization_x = "NonNegative", 
  regularization_y = "NonNegative",
  gamma_x = 0.5,
  gamma_y = 0.5,
  transform = "STANDARDIZE", 
  max_iterations = 2000,
  seed = 123
)

# Show predicted values
predict(k8_glrm_regularized, my_basket.h2o)[1:5, 1:5]
##   reconstr_7up reconstr_lasagna reconstr_pepsi reconstr_yop reconstr_red-wine
## 1     0.000000                0      0.0000000    0.0000000         0.0000000
## 2     0.000000                0      0.0000000    0.0000000         0.0000000
## 3     0.000000                0      0.0000000    0.0000000         0.0000000
## 4     0.609656                0      0.6311428    0.4565658         0.6697422
## 5     0.000000                0      0.0000000    0.0000000         0.8257210
## 
## [5 rows x 5 columns]

# Compare regularized versus non-regularized loss
par(mfrow = c(1, 2))
plot(k8_glrm)
plot(k8_glrm_regularized)

# Split data into train & validation
split <- h2o.splitFrame(my_basket.h2o, ratios = 0.75, seed = 123)
train <- split[[1]]
valid <- split[[2]]

# Create hyperparameter search grid
params <- expand.grid(
  regularization_x = c("None", "NonNegative", "L1"),
  regularization_y = c("None", "NonNegative", "L1"),
  gamma_x = seq(0, 1, by = .25),
  gamma_y = seq(0, 1, by = .25),
  error = 0,
  stringsAsFactors = FALSE
)

# Perform grid search
for(i in seq_len(nrow(params))) {
  
  # Create model
  glrm_model <- h2o.glrm(
    training_frame = train,
    k = 8, 
    loss = "Quadratic",
    regularization_x = params$regularization_x[i], 
    regularization_y = params$regularization_y[i],
    gamma_x = params$gamma_x[i],
    gamma_y = params$gamma_y[i],
    transform = "STANDARDIZE", 
    max_runtime_secs = 1000,
    seed = 123
  )
  
  # Predict on validation set and extract error
  validate <- h2o.performance(glrm_model, valid)
  params$error[i] <- validate@metrics$numerr
}

# Look at the top 10 models with the lowest error rate
params %>%
  arrange(error) %>%
  head(10)
##    regularization_x regularization_y gamma_x gamma_y    error
## 1                L1      NonNegative    1.00    0.25 13731.81
## 2                L1      NonNegative    1.00    0.50 13731.81
## 3                L1      NonNegative    1.00    0.75 13731.81
## 4                L1      NonNegative    1.00    1.00 13731.81
## 5                L1      NonNegative    0.75    0.25 13746.77
## 6                L1      NonNegative    0.75    0.50 13746.77
## 7                L1      NonNegative    0.75    0.75 13746.77
## 8                L1      NonNegative    0.75    1.00 13746.77
## 9                L1             None    0.75    0.00 13750.79
## 10               L1               L1    0.75    0.00 13750.79

# Apply final model with optimal hyperparamters
final_glrm_model <- h2o.glrm(
  training_frame = my_basket.h2o,
  k = 8, 
  loss = "Quadratic",
  regularization_x = "L1", 
  regularization_y = "NonNegative",
  gamma_x = 1,
  gamma_y = 0.25,
  transform = "STANDARDIZE", 
  max_iterations = 2000,
  seed = 123
)

# New observations to score
new_observations <- as.h2o(sample_n(my_basket, 2))

# Basic scoring
predict(final_glrm_model, new_observations) %>% round(0)
##   reconstr_7up reconstr_lasagna reconstr_pepsi reconstr_yop reconstr_red-wine reconstr_cheese reconstr_bbq reconstr_bulmers reconstr_mayonnaise reconstr_horlics reconstr_chicken-tikka reconstr_milk reconstr_mars reconstr_coke
## 1            0                0              0            0                 0               0            0                0                   0                0                      0             0             0             0
## 2            0               -1              1            0                 0               0            0                0                   0                1                      0             1             0             1
##   reconstr_lottery reconstr_bread reconstr_pizza reconstr_sunny-delight reconstr_ham reconstr_lettuce reconstr_kronenbourg reconstr_leeks reconstr_fanta reconstr_tea reconstr_whiskey reconstr_peas reconstr_newspaper
## 1                0              0              0                      0            0                0                    0              0              0            0                0             0                  0
## 2                0              0             -1                      0            0                0                    0              0              0            1                0             0                  0
##   reconstr_muesli reconstr_white-wine reconstr_carrots reconstr_spinach reconstr_pate reconstr_instant-coffee reconstr_twix reconstr_potatoes reconstr_fosters reconstr_soup reconstr_toad-in-hole reconstr_coco-pops
## 1               0                   0                0                0             0                       0             0                 0                0             0                     0                  0
## 2               1                   0                0                0             0                       1             0                 0                0             0                     0                  1
##   reconstr_kitkat reconstr_broccoli reconstr_cigarettes
## 1               0                 0                   0
## 2               0                 0                   0