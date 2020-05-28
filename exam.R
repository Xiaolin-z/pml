library(caret)
library(lattice)
library(ggplot2)
library(readr)
library(dplyr)
library(janitor)
library(rattle)
library(tibble)
library(bitops)
library(randomForest)
#input data
pml_testing <-read.csv("E:/法国高商/econometics/pml-testing.csv",stringsAsFactors = F)
pml_training<-read.csv("E:/法国高商/econometics/pml-training.csv",stringsAsFactors = F)

#check the variables and observations
dim(pml_training);dim(pml_testing)

#delete the basic information columns
pml_training<-pml_training[,-c(1:7)]

#delet all columns with NA
newtraining<-pml_training %>% select_if(~ !any(is.na(.)))

#delet all colums with ""
new1<-newtraining[!sapply(newtraining,function(x) any(x ==""))]

#set training and testing data set for model
set.seed(13234)
inBuild<-createDataPartition(y=new1$classe,
                           p=0.7,list=FALSE)
datatrain<-new1[inBuild,]
validation<-new1[-inBuild,]
inTrain<-createDataPartition(y=datatrain$classe,
                             p=0.7,list=FALSE)
training<-datatrain[inTrain,]
testing<-datatrain[-inTrain,]
dim(training);dim(testing);dim(validation)

#classification
table(training$classe)

#removing zero covatiates
nzv<-nearZeroVar(training,saveMetrics = TRUE)
delete<-nzv[nzv[,"nzv"]+nzv[,"zeroVar"]>0,]
dim(delete)

#correlation
M<-abs(cor(training[,-53]))
diag(M)<-0
a<-which(M>0.99,arr.ind = T)
dim(a)
head(a)
qplot(training$accel_belt_z,training$roll_belt,colour=training$classe)

#principal components
PC<-prcomp(training[,-53])
PC$rotation
summary(PC)


#PCA
preProc<-preProcess(training[,-53],method="pca")
trainPC<-predict(preProc,training[-53])
summary(trainPC)

##classification tree model
modFittree<-train(classe~.,method="rpart",data=training)
modFittree
print(modFittree$finalModel)

#plot
plot(modFittree$finalModel,uniform = TRUE,
     main="classification Tree")
text(modFittree$finalModel,use.n = TRUE,all = TRUE,cex=0.8)
fancyRpartPlot(modFittree$finalModel)

predtree<-predict(modFittree,testing)
table(predtree,testing$classe)
plot(table(predtree,testing$classe))
confusionMatrix(table(predtree,testing$classe))

#validation
predtreeV<-predict(modFittree,validation)
confusionMatrix(table(predtreeV,validation$classe))

##bagging model
#modFitbagging<-train(classe~.,method="treebag",data=training)
#modFitbagging
#table(predict(modFitbagging,testing),testing$classe)
#qplot(predict(modFitbagging,testing),testing$classe)

#random forest model
modFitrf<-train(classe~.,data=training,method="rf",prox=TRUE,
                trControl=trainControl(method="cv"),number=3)
modFitrf$finalModel
getTree(modFitrf$finalModel)
predrf<-predict(modFitrf,testing)
table(predrf,testing$classe)
confusionMatrix(table(predrf,testing$classe))
#validation
predrfV<-predict(modFitrf,validation)
confusionMatrix(table(predrfV,validation$classe))

##boosting
modFitboosting<-train(classe~.,method="gbm",data=training,
                      verbose=FALSE)
print(modFitboosting)
predboosting<-predict(modFitboosting,testing)
table(predboosting,testing$classe)
modFitboosting$finalModel
confusionMatrix(table(predboosting,testing$classe))
#validation
predboostingV<-predict(modFitboosting,validation)
confusionMatrix(table(predboostingV,validation$classe))

##combined model
qplot(predboosting,predrf,colour=classe,data = testing)

predDF<-data.frame(predtree,predrf,predboosting,classe=testing$classe)
combModFit<-train(classe~.,method="gam",data=predDF)
combModFit
combModFit$finalModel
combPred<-predict(combModFit,predDF)
table(combPred,testing$classe)
confusionMatrix(table(combPred,testing$classe))

#validation
predVDF<-data.frame(predtreeV,predboostingV,predrfV,validation$classe)
combPredV<-predict(combModFit,predVDF)
table(combPredV,validation$classe)
