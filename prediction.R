## ---- startprediction
load("preprocessed.RData")
library(reshape2)
library(gridExtra)
library(ggplot2)
library(data.table)
library(cvTools)
library(class)
library(nnet)
library(e1071)
library(pls)
library(rpart)
library(adabag)
library(randomForest)

set.seed(1)
## ---- knn
KNN <- function(xTrain, yTrain, xTest, nNeighbours)
{
  require(class)
  errKNNcv <- numeric(nNeighbours)
  # Use leave one out CV of train set to select number of neighbours
  for (K in seq_len(nNeighbours))
  {
    KNNpred <- knn.cv(train = xTrain, cl = yTrain, k = K)
    errKNNcv[K] <- sum(as.integer(KNNpred != yTrain))
  }
  # The simplest model is the one with the most neighbours
  K <- max(which(errKNNcv == min(errKNNcv))) 
  
  KNNpred <- knn(train = xTrain, test = xTest, cl = yTrain, k = K)
  print(K)
  
  KNNpred
}

## ---- logreg
logisticRegression <- function(xTrain, yTrain, xTest)
{
  require(nnet)
  maxVal <- max(abs(xTrain))
  dat <- data.frame(y = yTrain, unclass(xTrain/maxVal))
  model <- multinom(y ~ ., data = dat)
  logitPred <- predict(model, newdata= data.frame(xTest/maxVal), type='probs')
  
  logitPred <- factor(apply(logitPred, 1, which.max) - 1, levels = c(0:3))
  
  logitPred
}

## ---- svm
SVM <- function(xTrain, yTrain, xTest)
{
  require(e1071)
  dat <- data.frame(y = yTrain, unclass(xTrain))
  colnames(dat) <- c('y', colnames(xTrain))
  model <- svm(formula = y ~ ., data = dat, kernel = "linear")
  
  svmPred <- predict(model, xTest)
  
  svmPred
}

## ---- cart
cart <- function(xTrain, yTrain, xTest)
{
  require(rpart)
  dat <- data.frame(y = yTrain, unclass(xTrain))
  model <- rpart(y ~ ., data = dat, method = "class")
  pfit<- prune(model, cp= model$cptable[which.min(model$cptable[,"xerror"]),"CP"])
  
  cartPred <- predict(pfit, newdata = data.frame(xTest), type = c("class"))
  
  cartPred
}

## ---- boosting
boost <- function(xTrain, yTrain, xTest)
{
  require(adabag)
  dat <- data.frame(y = yTrain, unclass(xTrain))
  model <- boosting(y ~ ., data = dat)

  boostPred <- predict(model, newdata = data.frame(xTest))$class
  
  boostPred
}

## ---- randomForest
rnForest <- function(xTrain, yTrain, xTest)
{
  require(randomForest)
  dat <- data.frame(y = yTrain, unclass(xTrain))
  model <- randomForest(y ~ ., data = dat)
  
  randomForestPred <- predict(model, newdata = data.frame(xTest))
  
  randomForestPred
}



## ---- vote
doTest <- function(dataObject, f, ...)
{
  name <- as.character(substitute(f))
  res <- with(dataObject$data, f(xTrain, yTrain, xTest, ...))
  dataObject$predictions[[name]] <- res
  dataObject
}

majorityVote <- function(dataObject)
{
  dataObject$predictions$majorityVote <- apply(
    dataObject$predictions, 1, function(x)names(which.max(table(x))))
  dataObject
}

getErrorRates <- function(dataObject)
{
  yTest <- dataObject$data$yTest
  nTest <- length(yTest)
  dataObject$errorRate <- apply(dataObject$predictions, 2, function(x){sum(x != yTest)/nTest})
  dataObject
}

getMeanErrorRates <- function(dataList)
{
  err <- do.call('rbind', lapply(dataList, '[[', 'errorRate'))
  err <- colMeans(err)
  err
}

dataList <- lapply(dataList, doTest, KNN, nNeighbours = 20)
dataList <- lapply(dataList, doTest, logisticRegression)
dataList <- lapply(dataList, doTest, SVM)
dataList <- lapply(dataList, doTest, cart)
dataList <- lapply(dataList, doTest, boost)
dataList <- lapply(dataList, doTest, rnForest)
dataList <- lapply(dataList, majorityVote)

dataList <- lapply(dataList, getErrorRates)

getMeanErrorRates(dataList)

## ---- meanerror
df  = xtable(as.table(getMeanErrorRates(dataList)))
colnames(df) = c("error_test")
caption(df) = "Mean error rates for different models over 10-fold cross validation.\\label{table:meanerror}"
print(df)
# calculate standard error for test set
dataList2 <- lapply(dataList, majorityVote)
err2 <- do.call('rbind', lapply(dataList, '[[', 'errorRate'))
se <- colSums((err2 - matrix(rep(colMeans(err2),10),nrow(err2),ncol(err2),byrow=TRUE))^2)/(9*sqrt(10))
se <- xtable(as.table(se))
digits(se) <- 10
colnames(se) = c("stderror_test")
caption(se) = "Standard errors for different models over 10-fold cross validation.\\label{table:stderror}"
print(se)

## ---- visualizations
set.seed(1)
load("Case1_tst.RData")

testObject <- doTest(testObject, KNN, nNeighbours = 20)
testObject <- doTest(testObject, logisticRegression)
testObject <- doTest(testObject, SVM)
testObject <- doTest(testObject, cart)
testObject <- doTest(testObject, boost)
testObject <- doTest(testObject, rnForest)
testObject <- majorityVote(testObject)

pD <- data.frame(unclass(testObject$data$xTrain), class = testObject$data$yTrain, type = 'Train')
pD <- rbind(pD, data.frame(unclass(testObject$data$xTest), class = testObject$predictions$majorityVote, type = 'Test'))
colnames(pD) <- c('1', '2', '3', '4', 'class', 'type')


reshaper <- function(ind1, ind2, finalData)
{
  tmp <- pD[, c(ind1, ind2, 'class', 'type')]
  colnames(tmp) <- c('x', 'y', 'class', 'type')
  tmp[['r']] <- ind2
  tmp[['c']] <- ind1
  rbind(finalData, tmp)
}

finalData <- data.table(x = numeric(), y = numeric(), class = character() , type = character(), r = character(), c = character())
finalData <- reshaper('1', '2', finalData)
finalData <- reshaper('1', '3', finalData)
finalData <- reshaper('1', '4', finalData)
finalData <- reshaper('2', '3', finalData)
finalData <- reshaper('2', '4', finalData)
finalData <- reshaper('3', '4', finalData)

plotBase <- ggplot(finalData, aes(x = x, y = y, colour = class, shape = type)) + geom_point() + 
  scale_shape_manual(name = "Type", values = c(Train = 19, Test = 4)) +
  scale_colour_manual(name = "Class", values = c('0' = "#e41a1c", '1' = "#377eb8", '2' = "#4daf4a", '3' = "#984ea3")) +
  theme_bw() +
  facet_grid(r ~ c) +
  theme(axis.title = element_blank())

pdf(file = "visualization.pdf") 
plotBase
dev.off()
