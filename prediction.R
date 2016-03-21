## ---- startprediction
load("preprocessed.RData")
library(reshape2)
library(data.table)
library(cvTools)
library(class)
library(nnet)
library(e1071)
library(pls)
library(rpart)

## ---- knn
KNNfn <- function(xTrain, yTrain, xTest, yTest, nNeighbours)
{
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
    errKNN <- sum(as.integer(KNNpred != yTest))
    names(errKNN) <- "KNN"
    
    print(K)
    
    list(errorRate = errKNN, predictions = KNNpred)
}

## ---- logreg
logisticRegressionFn <- function(xTrain, yTrain, xTest, yTest){
    maxVal <- max(abs(xTrain))
    dat <- data.frame(y = yTrain, unclass(xTrain/maxVal))
    model <- multinom(y ~ ., data = dat)
    logitPred <- predict(model, newdata= data.frame(xTest/maxVal), type='probs')
    
    logitPred <- factor(apply(logitPred, 1, which.max) - 1, levels = c(0:3))
    errLogit <- sum(logitPred != yTest) / length(yTest)
    names(errLogit) <- "LogisticRegression"
    
    list(errorRate = errLogit, predictions = logitPred)
}

## ---- svm
svmFn <- function(xTrain, yTrain, xTest, yTest){
    dat <- data.frame(y = yTrain, unclass(xTrain))
    colnames(dat) <- c('y', colnames(xTrain))
    model <- svm(formula = y ~ ., data = dat, kernel = "linear")
    
    svmPred <- predict(model, xTest)
    errSVM <- sum(svmPred != yTest) / length(yTest)
    names(errSVM) <- "SVM"
    
    list(errorRate = errSVM, predictions = svmPred)
}

## ---- cart
cartFn <- function(xTrain, yTrain, xTest, yTest)
{
    dat <- data.frame(y = yTrain, unclass(xTrain))
    model <- rpart(y ~ ., data = dat, method = "class")
    pfit<- prune(model, cp= model$cptable[which.min(model$cptable[,"xerror"]),"CP"])
    
    cartPred <- predict(pfit, newdata = data.frame(xTest), type = c("class"))
    errCart <- sum(cartPred != yTest) / length(yTest)
    names(errCart) <- "CART"
    
    list(errorRate = errCart, predictions = cartPred)
}

## ---- vote
doTest <- function(dataObject, f, ...)
{
    name = as.character(substitute(f))
    res <- with(dataObject$data, f(xTrain, yTrain, xTest, yTest, ...))
    dataObject$predictions[[name]] <- res$predictions
    dataObject$errorRate <- c(dataObject$errorRate, res$errorRate)
    dataObject
}

majorityVote <- function(dataObject)
{
    dataObject$predictions$majorityVote <- apply(
      dataObject$predictions, 1, function(x)names(which.max(table(x))))
    dataObject$errorRate["majorityVote"] <- sum(
      dataObject$predictions$majorityVote != dataObject$data$yTest)
    dataObject
}

getMeanErrorRates <- function(dataList)
{
    dataList <- lapply(dataList, majorityVote)

    err <- do.call('rbind', lapply(dataList, '[[', 'errorRate'))
    err <- colMeans(err)
    err
}

dataList <- lapply(dataList, doTest, KNNfn, nNeighbours = 20)
dataList <- lapply(dataList, doTest, logisticRegressionFn)
dataList <- lapply(dataList, doTest, svmFn)
dataList <- lapply(dataList, doTest, cartFn)


getMeanErrorRates(dataList)

## ---- meanerror

df  = xtable(as.table(getMeanErrorRates(dataList)))
caption(df) = "Mean error rates for different models.\\label{table:meanerror}"
print(df)
