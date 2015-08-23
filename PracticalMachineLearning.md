# PracticalMachineLeraning
This is a project in the course practical machine learning, from John Hopkins coursera.org. The aim of this project is to predict the manner of which a group of participants performed different exercises (This is in the classe variable). This will be done with machine learning and the data supplied by <http://groupware.les.inf.puc-rio.br/har> and downloaded from
<https://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv>. The covariates used for prediction describe features such as position,acceleration and pitch of various body parts.

First we start by loading and cleaning the data.

```r
library(caret)
```

```
## Loading required package: lattice
## Loading required package: ggplot2
```

```r
data <- read.csv("pml-training.csv")
cleandata <- function(data){
        
        isBad <- sapply(1:dim(data)[2],function(x){sum(is.na(data)[,x])>10 || levels(data[,x])[1]==""})
        isBad[is.na(isBad)]<- FALSE
        data <- data[,!isBad]
        data[,2] <- as.numeric(data[,2])
        data <- data[,-c(1,3:6)]
        return(data)
}
data <- cleandata(data)
```
This cleaning does remove a lot of data but as we will see later,that's OK. If one would want better accuracy for the final model it would probably pay to include more variables.

To be able to estimate the out of sample error rate we create a training and test set

```r
#creating a test data set to use for out of sample error rate
inTrain <- createDataPartition(data[,55],p=0.7,list=FALSE)
training <- data[inTrain,]
testing <- data[-inTrain,]
```

Now it's time to train our model. We will use generalized boosted regression models for this, with 5 fold cross validation repeated 3 times.

```r
#Here we tell the train function to use 5 fold crossvalidation, 3 times
Control <- trainControl(method = "repeatedcv",number=5,repeats=3)
modFit <- train(classe~.,method="gbm",data=training,trControl=Control,verbose=FALSE)
```

```
## Loading required package: gbm
## Loading required package: survival
## 
## Attaching package: 'survival'
## 
## The following object is masked from 'package:caret':
## 
##     cluster
## 
## Loading required package: splines
## Loading required package: parallel
## Loaded gbm 2.1.1
## Loading required package: plyr
```
And the result of this can be viewed below

```r
#Getting the predicted values
pred <- predict(modFit,testing[,-55])
```

```
## Loading required package: gbm
## Loading required package: survival
## 
## Attaching package: 'survival'
## 
## The following object is masked from 'package:caret':
## 
##     cluster
## 
## Loading required package: splines
## Loading required package: parallel
## Loaded gbm 2.1.1
## Loading required package: plyr
```

```r
#printing the statistics of the prediction on the testing set (i.e out of sample error)
confusionMatrix(pred,testing[,55])
```

```
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction    A    B    C    D    E
##          A 1671   13    0    0    0
##          B    2 1115   12    2    2
##          C    0   10 1014    8    3
##          D    1    1    0  954    9
##          E    0    0    0    0 1068
## 
## Overall Statistics
##                                           
##                Accuracy : 0.9893          
##                  95% CI : (0.9863, 0.9918)
##     No Information Rate : 0.2845          
##     P-Value [Acc > NIR] : < 2.2e-16       
##                                           
##                   Kappa : 0.9865          
##  Mcnemar's Test P-Value : NA              
## 
## Statistics by Class:
## 
##                      Class: A Class: B Class: C Class: D Class: E
## Sensitivity            0.9982   0.9789   0.9883   0.9896   0.9871
## Specificity            0.9969   0.9962   0.9957   0.9978   1.0000
## Pos Pred Value         0.9923   0.9841   0.9797   0.9886   1.0000
## Neg Pred Value         0.9993   0.9949   0.9975   0.9980   0.9971
## Prevalence             0.2845   0.1935   0.1743   0.1638   0.1839
## Detection Rate         0.2839   0.1895   0.1723   0.1621   0.1815
## Detection Prevalence   0.2862   0.1925   0.1759   0.1640   0.1815
## Balanced Accuracy      0.9976   0.9876   0.9920   0.9937   0.9935
```

Thus the out of sample error rate is

```r
cat(100*(1-sum(pred==testing[,55])/length(pred)), '%')
```

```
## 1.070518 %
```
