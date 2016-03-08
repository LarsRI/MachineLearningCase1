library(reshape2)
library(data.table)
library(cvTools)
library(class)
library(nnet)
library(e1071)
set.seed(1)

load("Case1.RData")

setup <- data.table(
    sensor = rep(rep(c("S1", "S2", "S3"), each = 4092), 2),
    freq = rep(1:4092, 6),
    part = rep(c("real", "imaginary"), each = 4092*3)
)
setup[, idx:=seq_len(nrow(setup))]

# Select the 5% most variable frequencies
variation <- apply(Xtr, 2, sd)
ind <- variation > quantile(variation, 0.95)

# Print them
setup[ind, table(list(sensor, freq))]
selectedFreqs <- setup[ind, sort(unique(freq))]

# Notice that they are located in "islands". Select these islands
lst <- split(selectedFreqs, cumsum(c(TRUE, diff(selectedFreqs) > 10)))
regions <- data.table(cbind(
                            unlist(lapply(lst, min)), 
                            unlist(lapply(lst, max))
                            )
                      )
regions[, V1:=V1 - 50]
regions[, V2:=V2 + 50]

setkeyv(setup, 'freq')
setup[regions, sort(idx)]

# Make an index for easy subsetting in the future
subIndices <- setup[J(regions$V1)]
subIndices[, idxEnd := setup[J(regions$V2), idx]]
subIndices <- subIndices[, unlist(Map(seq, idx, idxEnd)), 
                         by = c("sensor", "part")
                         ]
subIndices[, sensor:=as.integer(as.factor(sensor))]
subIndices[, part:=as.integer(as.factor(part))]
setkeyv(subIndices, c("sensor", "part"))

## By playing around with the data I noticed that Removing first class 2, then 
## class 0 and finaly splitting class 1 and 3 with PCA. 
pcaTestFn <- function(X, exclGrps, class_tr, plot = TRUE)
{
    ind <- !class_tr %in% exclGrps
    cl <- class_tr[ind]
    sub <- X[ind, ]
    X.pca <- prcomp(sub,
                    center = TRUE,
                    scale. = TRUE) 
    if (plot)
    {
        oldPar <- par('mfrow')
        par(mfrow = c(1, 2))
        plot(X.pca)
        plot(X.pca$x[, 1], X.pca$x[, 2], col = cl)
        par(mfrow = oldPar)
    }
    invisible(X.pca)
}
X <- Xtr[, subIndices$V1]
pcaTestFn(X, c(), class_tr)
pcaTestFn(X, c(2), class_tr)
pcaTestFn(X, c(0, 2), class_tr)

## This is overfitting. I should have removed some datapoints before I started
## studying the data. But done is done.
## Analysing the scree-plots for elbows leads me to select 3, 1 and 2 PC from
## the threee PCAs. These will be used in the cross-validated analysis below.
rotateData <- function(x, pca, inds)
{
    x <- scale(x, pca$center, pca$scale)
    x %*% pca$rotation[, inds]
}

nObs <- nrow(X)
nFolds <- 10
folds <- cvFolds(nObs, nFolds)

nNeighbours <- 20
errKNN <- matrix(numeric(), nFolds, nNeighbours)
errLogit <- numeric(nFolds)
errSVN <- numeric(nFolds)
errMajority <- numeric(nFolds)

PCs <- array(NA_real_, dim = c(6, nFolds, ncol(X)))


for (i in seq_len(folds$K))
{
    ind <- folds$which == i
    Xtrain <- X[!ind, , drop = FALSE]
    Xtest <- X[ind, , drop = FALSE]
    classTrain <- class_tr[!ind]
    classTest <- class_tr[ind]
    
    pca1 <- pcaTestFn(Xtrain, c(), classTrain, FALSE)
    pca2 <- pcaTestFn(Xtrain, c(2), classTrain, FALSE)
    pca3 <- pcaTestFn(Xtrain, c(0, 2), classTrain, FALSE)
    
    PCs[1:3, i, ] <- pca1$rotation[, 1:3]
    PCs[4, i, ] <- pca2$rotation[, 1]
    PCs[5:6, i, ] <- pca3$rotation[, 1:2]
    
    trainRotated <- cbind(
        rotateData(Xtrain, pca1, 1:3),
        rotateData(Xtrain, pca2, 1),
        rotateData(Xtrain, pca3, 1:2)
    )
    colnames(trainRotated) <- NULL
    
    testRotated <- cbind(
        rotateData(Xtest, pca1, 1:3),
        rotateData(Xtest, pca2, 1),
        rotateData(Xtest, pca3, 1:2)
    )
    colnames(testRotated) <- NULL
    # KNN classification
    for (K in seq_len(nNeighbours))
    {
        KNNpred <- knn(train = trainRotated, test = testRotated, cl = classTrain, k = K)
        errKNN[i, K] <- sum(as.integer(KNNpred != classTest))
    }
    errKNN[i, ] <- errKNN[i, ] / sum(ind)
    
    maxVal <- max(abs(c(testRotated, trainRotated)))
    
    # Logistic regression
    dat <- data.frame(y = classTrain, trainRotated/maxVal)
    model <- multinom(y ~ ., data = dat)
    logitPred <- predict(model, newdata= data.frame(testRotated/maxVal), type='probs')
    logitPred <- factor(apply(logitPred, 1, which.max) - 1, levels = c(0:3))
    errLogit[i] <- sum(logitPred != classTest) / length(classTest)
    
    # SVM
    model <- svm(formula = y ~ ., data = dat, kernel = "linear")
    svnPred <- predict(model, testRotated)
    errSVN[i] <- sum(svnPred != classTest) / length(classTest)
    
    # KNN with N = 1 for majority vote.
    KNNpred1 <- knn(train = trainRotated, test = testRotated, cl = classTrain, k = 1)
    KNNpred3 <- knn(train = trainRotated, test = testRotated, cl = classTrain, k = 3)
    
    majorityVote <- rbind(KNNpred1, KNNpred3, logitPred, svnPred) - 1
    majorityVote <- as.integer(apply(majorityVote, 2, function(x){names(sort(table(x),decreasing=TRUE))[1]}))
    
    errMajority[i] <- sum(majorityVote != classTest)/length(classTest)
}

errKNNMean <- colMeans(errKNN)
Imin <- which.min(errKNNMean)
errKNNMean[Imin]
plot(errKNNMean)
mean(errLogit)
mean(errSVN)
mean(errMajority)
