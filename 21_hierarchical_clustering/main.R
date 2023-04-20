# Helper packages
library(dplyr)       # for data manipulation
library(ggplot2)     # for data visualization

# Modeling packages
library(cluster)     # for general clustering algorithms
library(factoextra)  # for visualizing cluster results

ames_scale <- AmesHousing::make_ames() %>%
  select_if(is.numeric) %>%  # select numeric columns
  select(-Sale_Price) %>%    # remove target column
  mutate_all(as.double) %>%  # coerce to double type
  scale()                    # center & scale the resulting columns

### hc clustering

# For reproducibility
set.seed(123)

# Dissimilarity matrix
d <- dist(ames_scale, method = "euclidean")

# Hierarchical clustering using Complete Linkage
hc1 <- hclust(d, method = "complete" )

# For reproducibility
set.seed(123)

# Compute maximum or complete linkage clustering with agnes
hc2 <- agnes(ames_scale, method = "complete")

# Agglomerative coefficient
hc2$ac
## [1] 0.926775

# methods to assess
m <- c( "average", "single", "complete", "ward")
names(m) <- c( "average", "single", "complete", "ward")

# function to compute coefficient
ac <- function(x) {
  agnes(ames_scale, method = x)$ac
}

# get agglomerative coefficient for each linkage method
purrr::map_dbl(m, ac)
##   average    single  complete      ward 
## 0.9139303 0.8712890 0.9267750 0.9766577

# compute divisive hierarchical clustering
hc4 <- diana(ames_scale)

# Divise coefficient; amount of clustering structure found
hc4$dc
## [1] 0.9191094

### determining optmial clusters
# Plot cluster results
p1 <- fviz_nbclust(ames_scale, FUN = hcut, method = "wss", 
                   k.max = 10) +
  ggtitle("(A) Elbow method")
p2 <- fviz_nbclust(ames_scale, FUN = hcut, method = "silhouette", 
                   k.max = 10) +
  ggtitle("(B) Silhouette method")
p3 <- fviz_nbclust(ames_scale, FUN = hcut, method = "gap_stat", 
                   k.max = 10) +
  ggtitle("(C) Gap statistic")

# Display plots side by side
gridExtra::grid.arrange(p1, p2, p3, nrow = 1)

### working with dendograms
# Construct dendorgram for the Ames housing example
hc5 <- hclust(d, method = "ward.D2" )
dend_plot <- fviz_dend(hc5)
dend_data <- attr(dend_plot, "dendrogram")
dend_cuts <- cut(dend_data, h = 8)
fviz_dend(dend_cuts$lower[[2]])

# Ward's method
hc5 <- hclust(d, method = "ward.D2" )

# Cut tree into 4 groups
sub_grp <- cutree(hc5, k = 8)

# Number of members in each cluster
table(sub_grp)
## sub_grp
##    1    2    3    4    5    6    7    8 
## 1363  567  650   36  123  156   24   11

# Plot full dendogram
fviz_dend(
  hc5,
  k = 8,
  horiz = TRUE,
  rect = TRUE,
  rect_fill = TRUE,
  rect_border = "jco",
  k_colors = "jco",
  cex = 0.1
)

dend_plot <- fviz_dend(hc5)                # create full dendogram
dend_data <- attr(dend_plot, "dendrogram") # extract plot info
dend_cuts <- cut(dend_data, h = 70.5)      # cut the dendogram at 
# designated height
# Create sub dendrogram plots
p1 <- fviz_dend(dend_cuts$lower[[1]])
p2 <- fviz_dend(dend_cuts$lower[[1]], type = 'circular')

# Side by side plots
gridExtra::grid.arrange(p1, p2, nrow = 1)

