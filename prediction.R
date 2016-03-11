load("preprocessed.RData")

KNNfn <- function(xTrain, yTrain, xTest, yTest, nNeighbours)
{
    errKNN <- numeric(nNeighbours)
    for (K in seq_len(nNeighbours))
    {
        KNNpred <- knn(train = xTrain, test = xTest, cl = yTrain, k = K)
        errKNN[K] <- sum(as.integer(KNNpred != yTest))
    }
    errKNN <- errKNN / length(yTest)
    names(errKNN) <- paste0("KNN_", seq_len(nNeighbours))
    
    K <- which.min(errKNN)
    KNNpred <- knn(train = xTrain, test = xTest, cl = yTrain, k = K)
    
    list(errorRate = errKNN, predictions = KNNpred)
}

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

svmFn <- function(xTrain, yTrain, xTest, yTest){
    dat <- data.frame(y = yTrain, unclass(xTrain))
    colnames(dat) <- c('y', colnames(xTrain))
    model <- svm(formula = y ~ ., data = dat, kernel = "linear")
    
    svmPred <- predict(model, xTest)
    errSVM <- sum(svmPred != yTest) / length(yTest)
    names(errSVM) <- "SVM"
    
    list(errorRate = errSVM, predictions = svmPred)
}

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
    dataObject$predictions$majorityVote <- apply(dataObject$predictions, 1, function(x)names(which.max(table(x))))
    dataObject$errorRate["majorityVote"] <- sum(dataObject$predictions$majorityVote != dataObject$data$yTest)
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

getMeanErrorRates(dataList)
