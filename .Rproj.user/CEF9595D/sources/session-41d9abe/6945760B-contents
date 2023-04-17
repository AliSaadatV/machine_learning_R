# Helper packages
library(dplyr)    # for data wrangling
library(ggplot2)  # for awesome graphics
library(rsample)  # for data splitting

# Modeling packages
library(caret)    # for classification and regression training
library(kernlab)  # for fitting SVMs

# Model interpretability packages
library(pdp)      # for partial dependence plots, etc.
library(vip)      # for variable importance plot

# Load attrition data
df <- attrition %>% 
  mutate_if(is.ordered, factor, ordered = FALSE)

# Create training (70%) and test (30%) sets
set.seed(123)  # for reproducibility
churn_split <- initial_split(df, prop = 0.7, strata = "Attrition")
churn_train <- training(churn_split)
churn_test  <- testing(churn_split)

### see parameters of different kernels
# Linear (i.e., soft margin classifier)
caret::getModelInfo("svmLinear")$svmLinear$parameters
##   parameter   class label
## 1         C numeric  Cost

# Polynomial kernel
caret::getModelInfo("svmPoly")$svmPoly$parameters
##   parameter   class             label
## 1    degree numeric Polynomial Degree
## 2     scale numeric             Scale
## 3         C numeric              Cost

# Radial basis kernel
caret::getModelInfo("svmRadial")$svmRadial$parameters
##   parameter   class label
## 1     sigma numeric Sigma
## 2         C numeric  Cost

### radial SVM
# Tune an SVM with radial basis kernel
set.seed(1854)  # for reproducibility
churn_svm <- train(
  Attrition ~ ., 
  data = churn_train,
  method = "svmRadial",               
  preProcess = c("center", "scale"),  
  trControl = trainControl(method = "cv", number = 10),
  tuneLength = 10
)

# Print results
churn_svm$results
##          sigma      C  Accuracy     Kappa  AccuracySD   KappaSD
## 1  0.009590249   0.25 0.8388542 0.0000000 0.004089627 0.0000000
## 2  0.009590249   0.50 0.8388542 0.0000000 0.004089627 0.0000000
## 3  0.009590249   1.00 0.8515233 0.1300469 0.014427649 0.1013069
## 4  0.009590249   2.00 0.8708857 0.3526368 0.023749215 0.1449342
## 5  0.009590249   4.00 0.8709611 0.4172884 0.026640331 0.1302496
## 6  0.009590249   8.00 0.8660873 0.4242800 0.026271496 0.1206188
## 7  0.009590249  16.00 0.8563495 0.4012563 0.026866012 0.1298460
## 8  0.009590249  32.00 0.8515138 0.3831775 0.028717623 0.1338717
## 9  0.009590249  64.00 0.8515138 0.3831775 0.028717623 0.1338717
## 10 0.009590249 128.00 0.8515138 0.3831775 0.028717623 0.1338717

### class weights (when one class is more important OR in case of imbalanced data)
class.weights = c("No" = 1, "Yes" = 10)

### class probabilities (used for AUC and ROC)
# Control params for SVM
ctrl <- trainControl(
  method = "cv", 
  number = 10, 
  classProbs = TRUE,                 
  summaryFunction = twoClassSummary  # also needed for AUC/ROC
)

# Tune an SVM
set.seed(5628)  # for reproducibility
churn_svm_auc <- train(
  Attrition ~ ., 
  data = churn_train,
  method = "svmRadial",               
  preProcess = c("center", "scale"),  
  metric = "ROC",  # area under ROC curve (AUC)       
  trControl = ctrl,
  tuneLength = 10
)

# Print results
churn_svm_auc$results
##          sigma      C       ROC      Sens      Spec      ROCSD      SensSD     SpecSD
## 1  0.009727585   0.25 0.8379109 0.9675488 0.3933824 0.06701067 0.012073306 0.11466031
## 2  0.009727585   0.50 0.8376397 0.9652767 0.3761029 0.06694554 0.010902039 0.14775214
## 3  0.009727585   1.00 0.8377081 0.9652633 0.4055147 0.06725101 0.007798768 0.09871169
## 4  0.009727585   2.00 0.8343294 0.9756750 0.3459559 0.06803483 0.012712528 0.14320366
## 5  0.009727585   4.00 0.8200427 0.9745255 0.3452206 0.07188838 0.013092221 0.12082675
## 6  0.009727585   8.00 0.8123546 0.9699278 0.3327206 0.07582032 0.013513393 0.11819788
## 7  0.009727585  16.00 0.7915612 0.9756883 0.2849265 0.07791598 0.010094292 0.10700782
## 8  0.009727585  32.00 0.7846566 0.9745255 0.2845588 0.07752526 0.010615423 0.08923723
## 9  0.009727585  64.00 0.7848594 0.9745255 0.2845588 0.07741087 0.010615423 0.09848550
## 10 0.009727585 128.00 0.7848594 0.9733895 0.2783088 0.07741087 0.010922892 0.10913126

confusionMatrix(churn_svm_auc)
## Cross-Validated (10 fold) Confusion Matrix 
## 
## (entries are percentual average cell counts across resamples)
##  
##           Reference
## Prediction   No  Yes
##        No  81.2  9.8
##        Yes  2.7  6.3
##                             
##  Accuracy (average) : 0.8748

### feature interpretation
prob_yes <- function(object, newdata) {
  predict(object, newdata = newdata, type = "prob")[, "Yes"]
}

# Variable importance plot
set.seed(2827)  # for reproducibility
vip(churn_svm_auc, method = "permute", nsim = 5, train = churn_train, 
    target = "Attrition", metric = "auc", reference_class = "Yes", 
    pred_wrapper = prob_yes)

features <- c("OverTime", "WorkLifeBalance", 
              "JobSatisfaction", "JobRole")
pdps <- lapply(features, function(x) {
  partial(churn_svm_auc, pred.var = x, which.class = 2,  
          prob = TRUE, plot = TRUE, plot.engine = "ggplot2") +
    coord_flip()
})
grid.arrange(grobs = pdps,  ncol = 2)