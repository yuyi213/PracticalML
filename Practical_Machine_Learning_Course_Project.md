# Practical Machine Learning Course Project
Yuyi Cheng  
March 27, 2016  


## Introduction
Using devices such as Jawbone Up, Nike FuelBand, and Fitbit it is now possible to collect a large amount of data about personal activity relatively inexpensively. These type of devices are part of the quantified self movement – a group of enthusiasts who take measurements about themselves regularly to improve their health, to find patterns in their behavior, or because they are tech geeks. One thing that people regularly do is quantify how much of a particular activity they do, but they rarely quantify how well they do it. In this project, our goal will be to use data from accelerometers on the belt, forearm, arm, and dumbell of 6 participants. They were asked to perform barbell lifts correctly and incorrectly in 5 different ways. More information is available from the website here: http://groupware.les.inf.puc-rio.br/har (see the section on the Weight Lifting Exercise Dataset). The data for this project come from this source: http://groupware.les.inf.puc-rio.br/har. 


## Data Processing
First, we need to load the relevant packages and to set up the global settings:

```r
library(knitr)
library(ggplot2)
library(lattice)
library(caret)
library(randomForest)
set.seed(0213)
opts_chunk$set(echo = TRUE, cache= TRUE)
```

Then, we can download the data files and load the data into R:

```r
if (!file.exists("pml-training.csv")){
        download.file("https://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv", "pml-training.csv")
}
pml_training <- read.csv("pml-training.csv")

if (!file.exists("pml-testing.csv")){
        download.file("https://d396qusza40orc.cloudfront.net/predmachlearn/pml-testing.csv", "pml-testing.csv")
}
pml_testing <- read.csv("pml-testing.csv")

dim1<- dim(pml_training)
dim2<- dim(pml_testing)
```

Now we can see the training dataset has 19622 observations of 160 variables, and the testing dataset has 20 observations of 160 variables. 

The distribution of the five different activities in the training dataset is:

```r
table(pml_training$classe)
```

```
## 
##    A    B    C    D    E 
## 5580 3797 3422 3216 3607
```

Now we'd like to reduce the noise of the training dataset since it has too many unusable variables. 


```r
pml_training <- pml_training[,-nearZeroVar(pml_training)]
pml_training <- pml_training[,-c(1:7)]
na_count <-data.frame(sapply(pml_training, function(x) sum(is.na(x))))
pml_training <- pml_training[,na_count<dim1[1]*0.6]
testing <- pml_testing[,colnames(pml_testing)%in%colnames(pml_training)]
dim3 <- dim(pml_training)
table(pml_training$classe)
```

```
## 
##    A    B    C    D    E 
## 5580 3797 3422 3216 3607
```
After removing unusable columns (very few unique values; user_name and time stamps; over 60% missing values), the dataset has 52 variables.  

## Training the Model
To start the modeling, we will need to partition the data for training and testing the models. We will use the 70% of the training dataset for training the model and 30% to test the model for accuracy estimate. And the original testing dataset from “pml-testing.csv” will be used to predict.


```r
inTrain <- createDataPartition(y=pml_training$classe, p=0.7, list=FALSE)
training <- pml_training[inTrain,]
validating <- pml_training[-inTrain,]
```

Now we have two datasets: training and validating. We can conduct random forrest algorithm to initiate the model to predict the activity category (classe variable in the dataset). Since we still have 52 predictors in our dataset, here we can use pca method to continue the data reducation, and then use pca data to construct the random forrest model. 


```r
pp <- preProcess(training[,-52], method = 'pca')
training_pca <- predict(pp, training[,-52])
validating_pca <- predict(pp, validating[,-52])
bestmtry <- tuneRF(training_pca,training$classe, ntreeTry=100, stepFactor=1.5,improve=0.01, trace=TRUE, plot=TRUE, dobest=FALSE)
```

```
## mtry = 5  OOB error = 3.15% 
## Searching left ...
## mtry = 4 	OOB error = 3.16% 
## -0.002309469 0.01 
## Searching right ...
## mtry = 7 	OOB error = 3.13% 
## 0.006928406 0.01
```

![](Practical_Machine_Learning_Course_Project_files/figure-html/unnamed-chunk-5-1.png) 

```r
mtry <- bestmtry[as.numeric(which.min(bestmtry[,"OOBError"])),"mtry"]
modelFit <- train(training$classe ~ ., method="rf", data=training_pca, ntree = 100, tuneGrid=data.frame(.mtry = mtry))
modelFit$finalModel
```

```
## 
## Call:
##  randomForest(x = x, y = y, ntree = 100, mtry = param$mtry) 
##                Type of random forest: classification
##                      Number of trees: 100
## No. of variables tried at each split: 7
## 
##         OOB estimate of  error rate: 3.05%
## Confusion matrix:
##      A    B    C    D    E class.error
## A 3861   17   16   11    1  0.01152074
## B   52 2543   54    3    6  0.04326561
## C    5   41 2315   30    5  0.03380634
## D    6    4   99 2135    8  0.05195382
## E    2   16   24   19 2464  0.02415842
```

## Validating the Model

Now we can evaluate our model with the validating dataset. 

```r
modelPredict <- predict(modelFit, validating_pca)
confusionMatrix(validating$classe,modelPredict)
```

```
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction    A    B    C    D    E
##          A 1662    3    7    1    1
##          B   19 1102   14    1    3
##          C    3   20  990   10    3
##          D    0    1   48  913    2
##          E    0    7   10    7 1058
## 
## Overall Statistics
##                                           
##                Accuracy : 0.9728          
##                  95% CI : (0.9683, 0.9768)
##     No Information Rate : 0.2862          
##     P-Value [Acc > NIR] : < 2.2e-16       
##                                           
##                   Kappa : 0.9656          
##  Mcnemar's Test P-Value : 3.53e-07        
## 
## Statistics by Class:
## 
##                      Class: A Class: B Class: C Class: D Class: E
## Sensitivity            0.9869   0.9726   0.9261   0.9796   0.9916
## Specificity            0.9971   0.9922   0.9925   0.9897   0.9950
## Pos Pred Value         0.9928   0.9675   0.9649   0.9471   0.9778
## Neg Pred Value         0.9948   0.9935   0.9837   0.9961   0.9981
## Prevalence             0.2862   0.1925   0.1816   0.1584   0.1813
## Detection Rate         0.2824   0.1873   0.1682   0.1551   0.1798
## Detection Prevalence   0.2845   0.1935   0.1743   0.1638   0.1839
## Balanced Accuracy      0.9920   0.9824   0.9593   0.9847   0.9933
```

## Testing the Model

With a model with 97% accuracy, we can now apply it to the given testing dataset to generate the final results:


```r
testing_pca <- predict(pp, testing)
testPredict <- predict(modelFit, testing_pca)
testPredict
```

```
##  [1] B A A A A E D B A A A C B A E E A B B B
## Levels: A B C D E
```


