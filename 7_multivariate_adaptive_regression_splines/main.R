# Helper packages
library(dplyr)     # for data wrangling
library(ggplot2)   # for awesome plotting

# Modeling packages
library(earth)     # for fitting MARS models
library(caret)     # for automating the tuning process

# Model interpretability packages
library(vip)       # for variable importance
library(pdp)       # for variable relationships

ames <- AmesHousing::make_ames()
set.seed(123)
split <- rsample::initial_split(ames, prop=0.7, strata="Sale_Price")
ames_train <- rsample::training(split)
ames_test <- rsample::testing(split)

### Fit a basic MARS model
mars1 <- earth(
  Sale_Price ~ .,  
  data = ames_train   
)

# Print model summary
print(mars1)
## Selected 36 of 39 terms, and 27 of 307 predictors
## Termination condition: RSq changed by less than 0.001 at 39 terms
## Importance: Gr_Liv_Area, Year_Built, Total_Bsmt_SF, ...
## Number of terms at each degree of interaction: 1 35 (additive model)
## GCV 557038757    RSS 1.065869e+12    GRSq 0.9136059    RSq 0.9193997

plot(mars1, which = 1)

### MARS with interactions
# Fit a basic MARS model
mars2 <- earth(
  Sale_Price ~ .,  
  data = ames_train,
  degree = 2
)

# check out the first 10 coefficient terms
summary(mars2) %>% .$coefficients %>% head(10)
##                                             Sale_Price
## (Intercept)                               2.331420e+05
## h(Gr_Liv_Area-2787)                       1.084015e+02
## h(2787-Gr_Liv_Area)                      -6.178182e+01
## h(Year_Built-2004)                        8.088153e+03
## h(2004-Year_Built)                       -9.529436e+02
## h(Total_Bsmt_SF-1302)                     1.131967e+02
## h(1302-Total_Bsmt_SF)                    -4.083722e+01
## h(2004-Year_Built)*h(Total_Bsmt_SF-1330) -1.553894e+00
## h(2004-Year_Built)*h(1330-Total_Bsmt_SF)  1.983699e-01
## Condition_1PosN*h(Gr_Liv_Area-2787)      -4.020535e+02

### Tunning
# create a tuning grid
hyper_grid <- expand.grid(
  degree = 1:3, 
  nprune = seq(2, 100, length.out = 10) %>% floor()
)

head(hyper_grid)
##   degree nprune
## 1      1      2
## 2      2      2
## 3      3      2
## 4      1     12
## 5      2     12
## 6      3     12

### Variable importance
# variable importance plots
p1 <- vip(cv_mars, num_features = 40, geom = "point", value = "gcv") + ggtitle("GCV")
p2 <- vip(cv_mars, num_features = 40, geom = "point", value = "rss") + ggtitle("RSS")

gridExtra::grid.arrange(p1, p2, ncol = 2)

# Construct partial dependence plots
p1 <- partial(cv_mars, pred.var = "Gr_Liv_Area", grid.resolution = 10) %>% 
  autoplot()
p2 <- partial(cv_mars, pred.var = "Year_Built", grid.resolution = 10) %>% 
  autoplot()
p3 <- partial(cv_mars, pred.var = c("Gr_Liv_Area", "Year_Built"), 
              grid.resolution = 10) %>% 
  plotPartial(levelplot = FALSE, zlab = "yhat", drape = TRUE, colorkey = TRUE, 
              screen = list(z = -20, x = -60))

# Display plots side by side
gridExtra::grid.arrange(p1, p2, p3, ncol = 3)