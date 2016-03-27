## ---- preprocessing
library(reshape2)
library(data.table)
library(cvTools)
library(R.matlab)

load("Case1.RData")

set.seed(1)
nObs <- nrow(Xtr)
nFolds <- 10
folds <- cvFolds(nObs, nFolds)

subsetData <- function(i, x, y, folds)
{
    ind <- folds$which == i
    xTrain <- x[!ind, ]
    yTrain <- y[!ind]
    xTest <- x[ind, ]
    yTest <- y[ind]
    list(xTrain = xTrain,
         yTrain = yTrain,
         xTest = xTest,
         yTest = yTest)
}


rotateData <- function(xTrain, yTrain, xTest, yTest)
{
    # Center, but do no scale x
    xTrain <- scale(xTrain, scale = FALSE)
    xTest <- scale(xTest, center = attr(xTrain, "scaled:center"), scale = FALSE)
    
    # Reshape y to be a 4 column matrix and do PLS
    y <- as.data.frame(lapply(levels(yTrain), function(x) as.integer(yTrain == x)))
    pls <- plsr(as.matrix(y) ~ xTrain)
    
    # shuffle xTrain, do PLS again and
    # find the maximum explained variance by the shuffled data.
    xShuf <- apply(xTrain, 2, function(x)x[sample(length(x))])
    plsShuf <- plsr(as.matrix(y) ~ xShuf)
    cutOff <- max(explvar(plsShuf))
    
    nComp <- max(which(explvar(pls) > cutOff))
    pls2 <- plsr(as.matrix(y) ~ xTrain, ncomp = nComp)
    # Select only components with a higher explained variance
    xTrainRot <- pls2$scores
    xTestRot <- predict(pls2, xTest, type = "scores")
    
    list(xTrain = xTrainRot,
         yTrain = yTrain,
         xTest = xTestRot,
         yTest = yTest)
}

makeSkeleton <- function(i, x, y, folds)
{
    data <- subsetData(i, x, y, folds)
    list(
        i = i,
        data = with(data, rotateData(xTrain, yTrain, xTest, yTest)),
        predictions = data.frame(matrix(nrow = length(data$yTest), ncol = 0)),
        errorRate = numeric()
    )
}

dataList <- lapply(seq_len(nFolds), makeSkeleton, Xtr, class_tr, folds)
save(dataList, folds, file = "preprocessed.RData")


## ---- testData
dat <- readMat('Case1_tst.mat')
xFinal <- dat$Xt

testObject <- list()
testObject$predictions = data.frame(matrix(nrow = nrow(xFinal), ncol = 0))
testObject$data <- rotateData(Xtr, class_tr, xFinal, 'Unknown')
save(testObject, file = "Case1_tst.RData")
