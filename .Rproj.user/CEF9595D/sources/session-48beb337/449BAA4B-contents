# Helper packages
library(dplyr)       # for data manipulation
library(ggplot2)     # for data visualization
library(stringr)     # for string functionality

# Modeling packages
library(cluster)     # for general clustering algorithms
library(factoextra)  # for visualizing cluster results

### kmeans
features <- mnist$train$images

# Use k-means model with 10 centers and 10 random starts
mnist_clustering <- kmeans(features, centers = 10, nstart = 10)

# Print contents of the model output
str(mnist_clustering)
## List of 9
##  $ cluster     : int [1:60000] 5 9 3 8 10 7 4 5 4 6 ...
##  $ centers     : num [1:10, 1:784] 0 0 0 0 0 0 0 0 0 0 ...
##   ..- attr(*, "dimnames")=List of 2
##   .. ..$ : chr [1:10] "1" "2" "3" "4" ...
##   .. ..$ : NULL
##  $ totss       : num 205706725984
##  $ withinss    : num [1:10] 23123576673 14119007546 16438261395 7950166288 ...
##  $ tot.withinss: num 153017742761
##  $ betweenss   : num 52688983223
##  $ size        : int [1:10] 7786 5384 5380 5515 7051 6706 4634 5311 4971 7262
##  $ iter        : int 8
##  $ ifault      : int 0
##  - attr(*, "class")= chr "kmeans"

# Extract cluster centers
mnist_centers <- mnist_clustering$centers

# Plot typical cluster digits
par(mfrow = c(2, 5), mar=c(0.5, 0.5, 0.5, 0.5))
layout(matrix(seq_len(nrow(mnist_centers)), 2, 5, byrow = FALSE))
for(i in seq_len(nrow(mnist_centers))) {
  image(matrix(mnist_centers[i, ], 28, 28)[, 28:1], 
        col = gray.colors(12, rev = TRUE), xaxt="n", yaxt="n")
}

# Create mode function
mode_fun <- function(x){  
  which.max(tabulate(x))
}

mnist_comparison <- data.frame(
  cluster = mnist_clustering$cluster,
  actual = mnist$train$labels
) %>%
  group_by(cluster) %>%
  mutate(mode = mode_fun(actual)) %>%
  ungroup() %>%
  mutate_all(factor, levels = 0:9)

# Create confusion matrix and plot results
yardstick::conf_mat(
  mnist_comparison, 
  truth = actual, 
  estimate = mode
) %>%
  autoplot(type = 'heatmap')

### choose best K

fviz_nbclust(
  my_basket, 
  kmeans, 
  k.max = 25,
  method = "wss",
  diss = get_dist(my_basket, method = "spearman")
)

### clustering with mixed data
# Full ames data set --> recode ordinal variables to numeric
ames_full <- AmesHousing::make_ames() %>%
  mutate_if(str_detect(names(.), 'Qual|Cond|QC|Qu'), as.numeric)

# One-hot encode --> retain only the features and not sale price
full_rank  <- caret::dummyVars(Sale_Price ~ ., data = ames_full, 
                               fullRank = TRUE)
ames_1hot <- predict(full_rank, ames_full)

# Scale data
ames_1hot_scaled <- scale(ames_1hot)

# New dimensions
dim(ames_1hot_scaled)
## [1] 2930  240

set.seed(123)

fviz_nbclust(
  ames_1hot_scaled, 
  kmeans, 
  method = "wss", 
  k.max = 25, 
  verbose = FALSE
)

# Original data minus Sale_Price
ames_full <- AmesHousing::make_ames() %>% select(-Sale_Price)

# Compute Gower distance for original data
gower_dst <- daisy(ames_full, metric = "gower")

# You can supply the Gower distance matrix to several clustering algos
pam_gower <- pam(x = gower_dst, k = 8, diss = TRUE)
diana_gower <- diana(x = gower_dst, diss = TRUE)
agnes_gower <- agnes(x = gower_dst, diss = TRUE)

### alternative partitioning methods
fviz_nbclust(
  ames_1hot_scaled, 
  pam, 
  method = "wss", 
  k.max = 25, 
  verbose = FALSE
)

# k-means computation time on MNIST data
system.time(kmeans(features, centers = 10))
##    user  system elapsed 
## 230.875   4.659 237.404

# CLARA computation time on MNIST data
system.time(clara(features, k = 10))
##   user  system elapsed 
## 37.975   0.286  38.966


