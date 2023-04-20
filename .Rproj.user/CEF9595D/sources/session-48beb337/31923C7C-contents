# Helper packages
library(dplyr)    # for data manipulation
library(ggplot2)  # for data visualization

# Modeling packages
library(mclust)   # for fitting clustering algorithms

### measuring probability and uncertainty
# Apply GMM model with 3 components
geyser_mc <- Mclust(geyser, G = 3)

# Plot results
plot(geyser_mc, what = "density")
plot(geyser_mc, what = "uncertainty")

# Observations with high uncertainty
sort(geyser_mc$uncertainty, decreasing = TRUE) %>% head()
##       187       211        85       285        28       206 
## 0.4689087 0.4542588 0.4355496 0.4355496 0.4312406 0.4168440

### model selection
summary(geyser_mc)
## ---------------------------------------------------- 
## Gaussian finite mixture model fitted by EM algorithm 
## ---------------------------------------------------- 
## 
## Mclust EEI (diagonal, equal volume and shape) model with 3 components: 
## 
##  log-likelihood   n df      BIC       ICL
##       -1371.823 299 10 -2800.65 -2814.577
## 
## Clustering table:
##   1   2   3 
##  91 107 101

geyser_optimal_mc <- Mclust(geyser)

summary(geyser_optimal_mc)
## ---------------------------------------------------- 
## Gaussian finite mixture model fitted by EM algorithm 
## ---------------------------------------------------- 
## 
## Mclust VVI (diagonal, varying volume and shape) model with 4 components: 
## 
##  log-likelihood   n df       BIC       ICL
##        -1330.13 299 19 -2768.568 -2798.746
## 
## Clustering table:
##  1  2  3  4 
## 90 17 98 94

legend_args <- list(x = "bottomright", ncol = 5)
plot(geyser_optimal_mc, what = 'BIC', legendArgs = legend_args)
plot(geyser_optimal_mc, what = 'classification')
plot(geyser_optimal_mc, what = 'uncertainty')

### example
my_basket_mc <- Mclust(my_basket, 1:20)

summary(my_basket_mc)
## ---------------------------------------------------- 
## Gaussian finite mixture model fitted by EM algorithm 
## ---------------------------------------------------- 
## 
## Mclust EEV (ellipsoidal, equal volume and shape) model with 6 components: 
## 
##  log-likelihood    n   df      BIC       ICL
##        8308.915 2000 5465 -24921.1 -25038.38
## 
## Clustering table:
##   1   2   3   4   5   6 
## 391 403  75 315 365 451

plot(my_basket_mc, what = 'BIC', 
     legendArgs = list(x = "bottomright", ncol = 5))

probabilities <- my_basket_mc$z 
colnames(probabilities) <- paste0('C', 1:6)

probabilities <- probabilities %>%
  as.data.frame() %>%
  mutate(id = row_number()) %>%
  tidyr::gather(cluster, probability, -id)

ggplot(probabilities, aes(probability)) +
  geom_histogram() +
  facet_wrap(~ cluster, nrow = 2)

uncertainty <- data.frame(
  id = 1:nrow(my_basket),
  cluster = my_basket_mc$classification,
  uncertainty = my_basket_mc$uncertainty
)

uncertainty %>%
  group_by(cluster) %>%
  filter(uncertainty > 0.25) %>%
  ggplot(aes(uncertainty, reorder(id, uncertainty))) +
  geom_point() +
  facet_wrap(~ cluster, scales = 'free_y', nrow = 1)

cluster2 <- my_basket %>%
  scale() %>%
  as.data.frame() %>%
  mutate(cluster = my_basket_mc$classification) %>%
  filter(cluster == 2) %>%
  select(-cluster)

cluster2 %>%
  tidyr::gather(product, std_count) %>%
  group_by(product) %>%
  summarize(avg = mean(std_count)) %>%
  ggplot(aes(avg, reorder(product, avg))) +
  geom_point() +
  labs(x = "Average standardized consumption", y = NULL)
