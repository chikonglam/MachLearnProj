# Random Forest Is an Accurate Algorithm in Classifying Lifting Errors with Accelerometers
Chi Lam  
February 12, 2016  

Introduction
==============
The *Weight Lifting Exercises Dataset* of the *Human Activity Recognition* project were used to determine if a *Classification Tree* algorithm, or a *Random Forest* algorithm is more accurate to identify four types of errors (class B, C, D, E) from wearable accelerometer data (class A means the exercise was done correctly).     

Executive Summary
==================
A *Random Forest* algorithm is determined to be significantly more accurate to classify lifting exercise errors in the *Weight Lifting Exercises Dataset*.

Data Loading and Preparation
============================
### Libraries and Parameters
The caret library  is loaded for the machine learning capabilities, and a random number seed is set to ensure reproducibility.

```r
library(caret)
set.seed(852489)
```

### Loading the Datasets
The following code loads the training and testing datasets in.  The original data source is available on the Human Activity Recognition site:  http://groupware.les.inf.puc-rio.br/har 

```r
training <‐ read.csv(url("https://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv"), na.strings=c("NA","#DIV/0!",""))
testing <‐ read.csv(url("https://d396qusza40orc.cloudfront.net/predmachlearn/pml-testing.csv"), na.strings=c("NA","#DIV/0!",""))
```

### Preparing and cleaning the Data
The *training* dataset is split into two, one for training the models, one for evaluating the models.  The training part of the data is only 60% to keep the model training process speed at an acceptable level.  Variables that are mostly constant, NA, and otherwise not useful are removed to improve the performance of the models.

```r
#partition into training and testing
inTrain <- createDataPartition(y=training$classe, p=0.6, list=FALSE)
myTraining <- training[inTrain, ]
myTesting <- training[‐inTrain, ]

# remove variables with nearly zero variance
nzv <- nearZeroVar(myTraining)
myTraining <- myTraining[, -nzv]
myTesting <- myTesting[, -nzv]

# remove variables that are often NA
mostlyNA <- sapply(myTraining, function(x) mean(is.na(x))) > 0.80
myTraining <- myTraining[, mostlyNA==F]
myTesting <- myTesting[, mostlyNA==F]

# remove first five variables
myTraining <- myTraining[, -(1:5)]
myTesting <- myTesting[, -(1:5)]
```

Algorithm Selection: Classification Tree Vs. Random Forest
==============================
Classification Tree and Random Forest algorithms were used to build models to classify the exercise errors.

### Classification Tree 
The following code builds a *Classification Tree* model.  It also preprocesses the data by centering and scaling it to improve the model's performance, and limit the cross validation to 3 folds to make the model comparable to Random Forest. Random Forest needs this limit to have an acceptable speed.


```r
rPartFitControl <- trainControl(method="cv", number=3, verboseIter=F)
modRPart <- train(classe ~ ., data=myTraining, preProcess=c("center", "scale"), method="rpart", trControl=rPartFitControl)
```

The following code evaluates the accuracy of the model.

```r
predRPart <- predict(modRPart, myTesting)
conMatRPart <- confusionMatrix(predRPart, myTesting$classe)
accuracyRPart <- conMatRPart$overall["Accuracy"]; print(accuracyRPart)
```

```
##  Accuracy 
## 0.5655111
```
The *Classification Tree* only has an out-of-sample accuracy of 0.5655111.  It is therefore not a very good model to classify the lifting exercise errors.

### Random Forest
The following code builds a *Random Forest* model.  It limits the cross validation to 3 folds to have an acceptable speed. The Classification Tree model has this limit too to make both models comparable.  (Although already limited to give acceptable speed, the code still took about 15 minutes for a moderately fast computer in 2016 to complete)


```r
RfFitControl <- trainControl(method="cv", number=3, verboseIter=F)
modRf <- train(classe ~ ., data=myTraining, method="rf", trControl=RfFitControl)
```

The following code evaluates the accuracy of the model.


```r
predRf <- predict(modRf, myTesting)
conMatRf <- confusionMatrix(predRf, myTesting$classe)
accuracyRf <- conMatRf$overall["Accuracy"]; print(accuracyRf)
```

```
##  Accuracy 
## 0.9980882
```

The *Random Forest* has an out-of-sample accuracy of 0.9980882.  It is a very good accuracy, therefore, it is chosen to classify the lifting exercise errors of the testing data set.  The following code shows the confusion matrix of the Random Forest model: it shows the excellent accuracy in more details.

```r
print(conMatRf)
```

```
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction    A    B    C    D    E
##          A 2232    5    0    0    0
##          B    0 1512    2    0    0
##          C    0    0 1366    3    0
##          D    0    1    0 1283    4
##          E    0    0    0    0 1438
## 
## Overall Statistics
##                                           
##                Accuracy : 0.9981          
##                  95% CI : (0.9968, 0.9989)
##     No Information Rate : 0.2845          
##     P-Value [Acc > NIR] : < 2.2e-16       
##                                           
##                   Kappa : 0.9976          
##  Mcnemar's Test P-Value : NA              
## 
## Statistics by Class:
## 
##                      Class: A Class: B Class: C Class: D Class: E
## Sensitivity            1.0000   0.9960   0.9985   0.9977   0.9972
## Specificity            0.9991   0.9997   0.9995   0.9992   1.0000
## Pos Pred Value         0.9978   0.9987   0.9978   0.9961   1.0000
## Neg Pred Value         1.0000   0.9991   0.9997   0.9995   0.9994
## Prevalence             0.2845   0.1935   0.1744   0.1639   0.1838
## Detection Rate         0.2845   0.1927   0.1741   0.1635   0.1833
## Detection Prevalence   0.2851   0.1930   0.1745   0.1642   0.1833
## Balanced Accuracy      0.9996   0.9979   0.9990   0.9985   0.9986
```
Note that the 95% confidence interval of this out-of-sample accuracy is ( 0.9968487, 0.9989296 ), therefore, the estimated out-of-sample accuracy of another dataset is also estimated to be between 0.9968487 and 0.9989296.

Classifying Lifting Exercise Errors Using Random Forest
==========
The following code classifies the errors in the test data set using the Random Forest model.  The results were submitted to the Coursera test and were graded correct.

```r
predict(modRf, newdata=testing)
```

```
##  [1] B A B A A E D B A A B C B A E E A B B B
## Levels: A B C D E
```

Conclusion
==========
Using a *Random Forest* algorithm can accurately classify the error type in a weight lifting exercise with wearable accelerometer data.  The out-of-sample accuracy is 0.9980882.  It is significantly better than using a *Classification Tree* algorithm (out-of-sample accuracy is only 0.5655111).  The *Random Forest* algorithm takes significantly longer than the *Classification Tree* algorithm, but the huge accuracy difference justifies the added cost.

References
===========
Velloso, E.; Bulling, A.; Gellersen, H.; Ugulino, W.; Fuks, H. Qualitative Activity Recognition of Weight Lifting Exercises. Proceedings of 4th International Conference in Cooperation with SIGCHI (Augmented Human '13) . Stuttgart, Germany: ACM SIGCHI, 2013.
