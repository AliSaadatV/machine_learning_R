library(dplyr)       # basic data manipulation and plotting
library(ggplot2)     # data visualization
library(h2o)         # performing dimension reduction

url <- "https://koalaverse.github.io/homlr/data/my_basket.csv"
my_basket <- readr::read_csv(url)
dim(my_basket)  
## [1] 2000   42

h2o.no_progress()  # turn off progress bars for brevity
h2o.init(max_mem_size = "5g")  # connect to H2O instance

# convert data to h2o object
my_basket.h2o <- as.h2o(my_basket)

# run PCA
my_pca <- h2o.prcomp(
  training_frame = my_basket.h2o,
  pca_method = "GramSVD",
  k = ncol(my_basket.h2o), 
  transform = "STANDARDIZE", 
  impute_missing = TRUE,
  max_runtime_secs = 1000
)

my_pca
## Model Details:
## ==============
## 
## H2ODimReductionModel: pca
## Model ID:  PCA_model_R_1536152543598_1 
## Importance of components: 
##                             pc1      pc2      pc3      pc4      pc5      pc6      pc7      pc8      pc9
## Standard deviation     1.513919 1.473768 1.459114 1.440635 1.435279 1.411544 1.253307 1.026387 1.010238
## Proportion of Variance 0.054570 0.051714 0.050691 0.049415 0.049048 0.047439 0.037400 0.025083 0.024300
## Cumulative Proportion  0.054570 0.106284 0.156975 0.206390 0.255438 0.302878 0.340277 0.365360 0.389659
##                            pc10     pc11     pc12     pc13     pc14     pc15     pc16     pc17     pc18
## Standard deviation     1.007253 0.988724 0.985320 0.970453 0.964303 0.951610 0.947978 0.944826 0.932943
## Proportion of Variance 0.024156 0.023276 0.023116 0.022423 0.022140 0.021561 0.021397 0.021255 0.020723
## Cumulative Proportion  0.413816 0.437091 0.460207 0.482630 0.504770 0.526331 0.547728 0.568982 0.589706
##                            pc19     pc20     pc21     pc22     pc23     pc24     pc25     pc26     pc27
## Standard deviation     0.931745 0.924207 0.917106 0.908494 0.903247 0.898109 0.894277 0.876167 0.871809
## Proportion of Variance 0.020670 0.020337 0.020026 0.019651 0.019425 0.019205 0.019041 0.018278 0.018096
## Cumulative Proportion  0.610376 0.630713 0.650739 0.670390 0.689815 0.709020 0.728061 0.746339 0.764436
##                            pc28     pc29     pc30     pc31     pc32     pc33     pc34     pc35     pc36
## Standard deviation     0.865912 0.855036 0.845130 0.842818 0.837655 0.826422 0.818532 0.813796 0.804380
## Proportion of Variance 0.017852 0.017407 0.017006 0.016913 0.016706 0.016261 0.015952 0.015768 0.015405
## Cumulative Proportion  0.782288 0.799695 0.816701 0.833614 0.850320 0.866581 0.882534 0.898302 0.913707
##                            pc37     pc38     pc39     pc40     pc41     pc42
## Standard deviation     0.796073 0.793781 0.780615 0.778612 0.763433 0.749696
## Proportion of Variance 0.015089 0.015002 0.014509 0.014434 0.013877 0.013382
## Cumulative Proportion  0.928796 0.943798 0.958307 0.972741 0.986618 1.000000
## 
## 
## H2ODimReductionMetrics: pca
## 
## No model metrics available for PCA

# influence of each feature on PC1
my_pca@model$eigenvectors %>% 
  as.data.frame() %>% 
  mutate(feature = row.names(.)) %>%
  ggplot(aes(pc1, reorder(feature, pc1))) +
  geom_point()

# plot PC1 vs PC2
my_pca@model$eigenvectors %>% 
  as.data.frame() %>% 
  mutate(feature = row.names(.)) %>%
  ggplot(aes(pc1, pc2, label = feature)) +
  geom_text()

### Selecting number of PCs

### Eigenvalue criteria (keep all eigenvalue > 1)
# Compute eigenvalues
eigen <- my_pca@model$importance["Standard deviation", ] %>%
  as.vector() %>%
  .^2

# Sum of all eigenvalues equals number of variables
sum(eigen)
## [1] 42

# Find PCs where the sum of eigenvalues is greater than or equal to 1
which(eigen >= 1)
##  [1]  1  2  3  4  5  6  7  8  9 10

### proportion of variance explained
# Extract and plot PVE and CVE
data.frame(
  PC  = my_pca@model$importance %>% seq_along(),
  PVE = my_pca@model$importance %>% .[2,] %>% unlist(),
  CVE = my_pca@model$importance %>% .[3,] %>% unlist()
) %>%
  tidyr::gather(metric, variance_explained, -PC) %>%
  ggplot(aes(PC, variance_explained)) +
  geom_point() +
  facet_wrap(~ metric, ncol = 1, scales = "free")

# How many PCs required to explain at least 75% of total variability
min(which(ve$CVE >= 0.75))
## [1] 27

### scree plot
data.frame(
  PC  = my_pca@model$importance %>% seq_along,
  PVE = my_pca@model$importance %>% .[2,] %>% unlist()
) %>%
  ggplot(aes(PC, PVE, group = 1, label = PC)) +
  geom_point() +
  geom_line() +
  geom_text(nudge_y = -.002)
