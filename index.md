# Document Overview

This document is intended as a Peer-Reviewed Assignment from Coursera’s Practical Machine Learning course. As required, it is provided in HTML format.<br />

The main goal of the Assignment is to predict the manner in which 6 participants have performed a test - correctly or incorrectly (see: Background). This is the “classe” variable in the training data.<br />

The document provides a deliberation and structure for the chosen prediction model, using the R programming language. It continues with cross-validation and comments on the sample error.<br />

The resulting machine learning algorithm is applied to the 20 test cases available in the testing data. The predictions are published as a conclusion to this document (see: Predictions).<br />

# Background

> “One thing that people regularly do is quantify how much of a particular activity they do, but they rarely quantify how well they do it.” (From the Assignment)<br />

Devices such as Jawbone Up, Nike FuelBand, and Fitbit make it possible to collect a large amount of data about personal activity.<br />

In the Human Activity Recognition (HAR) project this Assignment is based on, we use data from accelerometers on the belt, forearm, arm, and dumbell of 6 test participants.<br />

The participants were asked to perform barbell lifts correctly and incorrectly in 5 different ways.<br />

# Environment

R Studio Version 1.0.44<br /><br />
R version 3.3.2 (2016-10-31) -- "Sincere Pumpkin Patch"<br />
Platform: x86_64-mingw32 (Windows 10), x86_64-apple-darwin13.4.0 (OS-X)<br />

R packages to be run:<br />

`library(caret)`<br />
`library(doParallel)`<br />
`library(dplyr)`<br />
`library(randomForest)`<br />

**Note:** ggplot2, lattice are caret package dependencies and will be loaded together with it.<br />

# Obtaining and Cleaning Data

For the purpose of this Assignment, we assume that the data is downloaded to our working directory in R.<br />

## Loading the Data

On reviewing the data, we see numerous “division by zero” (#DIV/0!) entries. In order to treat these and the remainder of the values as numeric, we will flag the “division by zero” entries as “NA”.<br />

We load the training and testing data sets to appropriately named data frame variables:<br />

`trainingPMLinit <- read.csv('pml-training.csv', na.strings=c("NA", "#DIV/0!"))`<br />
`testingPMLinit <- read.csv('pml-testing.csv', na.strings=c("NA", "#DIV/0!"))`<br />

We see they contain 160 variables each:<br />

`dim(trainingPMLinit)`<br />
`[1] 19622   160`<br />
`dim(testingPMLinit)`<br />
`[1]  20 160`<br />

## Disposing of Out of Scope Variables

Given the nature of the Assignment, we are focused on measurements and not on data related to who performed the assignment, and at what timespan.<br />

With that in mind, on running the names() function over the training and testing data frames, we see that we can do without the first 7 variables.<br />

<details>
  <summary>Click to expand</summary>
names(trainingPMLinit)<br />
  [1] "X"                        "user_name"                "raw_timestamp_part_1"     "raw_timestamp_part_2"  <br />  
  [5] "cvtd_timestamp"           "new_window"               "num_window"               "roll_belt"             <br /> 
  [9] "pitch_belt"               "yaw_belt"                 "total_accel_belt"         "kurtosis_roll_belt"      <br />
 [13] "kurtosis_picth_belt"      "kurtosis_yaw_belt"        "skewness_roll_belt"       "skewness_roll_belt.1"    <br />
 [17] "skewness_yaw_belt"        "max_roll_belt"            "max_picth_belt"           "max_yaw_belt"            <br />
 [21] "min_roll_belt"            "min_pitch_belt"           "min_yaw_belt"             "amplitude_roll_belt"     <br />
 [25] "amplitude_pitch_belt"     "amplitude_yaw_belt"       "var_total_accel_belt"     "avg_roll_belt"           <br />
 [29] "stddev_roll_belt"         "var_roll_belt"            "avg_pitch_belt"           "stddev_pitch_belt"       <br />
 [33] "var_pitch_belt"           "avg_yaw_belt"             "stddev_yaw_belt"          "var_yaw_belt"            <br />
 [37] "gyros_belt_x"             "gyros_belt_y"             "gyros_belt_z"             "accel_belt_x"            <br />
 [41] "accel_belt_y"             "accel_belt_z"             "magnet_belt_x"            "magnet_belt_y"           <br />
 [45] "magnet_belt_z"            "roll_arm"                 "pitch_arm"                "yaw_arm"                 <br />
 [49] "total_accel_arm"          "var_accel_arm"            "avg_roll_arm"             "stddev_roll_arm"         <br />
 [53] "var_roll_arm"             "avg_pitch_arm"            "stddev_pitch_arm"         "var_pitch_arm"           <br />
 [57] "avg_yaw_arm"              "stddev_yaw_arm"           "var_yaw_arm"              "gyros_arm_x"             <br />
 [61] "gyros_arm_y"              "gyros_arm_z"              "accel_arm_x"              "accel_arm_y"             <br />
 [65] "accel_arm_z"              "magnet_arm_x"             "magnet_arm_y"             "magnet_arm_z"            <br />
 [69] "kurtosis_roll_arm"        "kurtosis_picth_arm"       "kurtosis_yaw_arm"         "skewness_roll_arm"       <br />
 [73] "skewness_pitch_arm"       "skewness_yaw_arm"         "max_roll_arm"             "max_picth_arm"           <br />
 [77] "max_yaw_arm"              "min_roll_arm"             "min_pitch_arm"            "min_yaw_arm"             <br />
 [81] "amplitude_roll_arm"       "amplitude_pitch_arm"      "amplitude_yaw_arm"        "roll_dumbbell"           <br />
 [85] "pitch_dumbbell"           "yaw_dumbbell"             "kurtosis_roll_dumbbell"   "kurtosis_picth_dumbbell" <br />
 [89] "kurtosis_yaw_dumbbell"    "skewness_roll_dumbbell"   "skewness_pitch_dumbbell"  "skewness_yaw_dumbbell"   <br />
 [93] "max_roll_dumbbell"        "max_picth_dumbbell"       "max_yaw_dumbbell"         "min_roll_dumbbell"       <br />
 [97] "min_pitch_dumbbell"       "min_yaw_dumbbell"         "amplitude_roll_dumbbell"  "amplitude_pitch_dumbbell"<br />
[101] "amplitude_yaw_dumbbell"   "total_accel_dumbbell"     "var_accel_dumbbell"       "avg_roll_dumbbell"       <br />
[105] "stddev_roll_dumbbell"     "var_roll_dumbbell"        "avg_pitch_dumbbell"       "stddev_pitch_dumbbell"   <br />
[109] "var_pitch_dumbbell"       "avg_yaw_dumbbell"         "stddev_yaw_dumbbell"      "var_yaw_dumbbell"        <br />
[113] "gyros_dumbbell_x"         "gyros_dumbbell_y"         "gyros_dumbbell_z"         "accel_dumbbell_x"        <br />
[117] "accel_dumbbell_y"         "accel_dumbbell_z"         "magnet_dumbbell_x"        "magnet_dumbbell_y"       <br />
[121] "magnet_dumbbell_z"        "roll_forearm"             "pitch_forearm"            "yaw_forearm"             <br />
[125] "kurtosis_roll_forearm"    "kurtosis_picth_forearm"   "kurtosis_yaw_forearm"     "skewness_roll_forearm"   <br />
[129] "skewness_pitch_forearm"   "skewness_yaw_forearm"     "max_roll_forearm"         "max_picth_forearm"       <br />
[133] "max_yaw_forearm"          "min_roll_forearm"         "min_pitch_forearm"        "min_yaw_forearm"         <br />
[137] "amplitude_roll_forearm"   "amplitude_pitch_forearm"  "amplitude_yaw_forearm"    "total_accel_forearm"     <br />
[141] "var_accel_forearm"        "avg_roll_forearm"         "stddev_roll_forearm"      "var_roll_forearm"        <br />
[145] "avg_pitch_forearm"        "stddev_pitch_forearm"     "var_pitch_forearm"        "avg_yaw_forearm"         <br />
[149] "stddev_yaw_forearm"       "var_yaw_forearm"          "gyros_forearm_x"          "gyros_forearm_y"         <br />
[153] "gyros_forearm_z"          "accel_forearm_x"          "accel_forearm_y"          "accel_forearm_z"         <br />
[157] "magnet_forearm_x"         "magnet_forearm_y"         "magnet_forearm_z"         "classe"  <br />
</details>

(Similarly for the testing data frame).<br />

`trainingPMLinit <- trainingPMLinit[ , -c(1:7)]`<br />
`testingPMLinit <- testingPMLinit[ , -c(1:7)]`<br />

## Disposing of Variables with NA Values

Getting back to the NA, we can as well clear out the variables that contain mostly NA. Let’s see how many columns contain all NA and how many - any NA:<br />

`trainingAllNA <- sapply(trainingPMLinit, function(x)all(is.na(x)))`<br />
`trainingAnyNA <- sapply(trainingPMLinit, function(x)any(is.na(x)))`<br />

`compareAllAnyNA <- mapply(setdiff, trainingAllNA, trainingAnyNA)`<br />
`num.compareAllAnyNA <- sapply(compareAllAnyNA, length)`<br />

Reviewing the training data set as well as the output (num.compareAllAnyNA), it would seem that almost all columns containing NA values are likely to be all NA. We can therefore perform the below clean-up.<br />

`trainingNA <- sapply(trainingPMLinit, function(x) mean(is.na(x))) > 0.95`<br />
`trainingPMLinit <- trainingPMLinit[ , !trainingNA]`<br />

Reviewing the testing data set, all columns containing NA contain nothing else (all NA).<br />

`testingNA <- apply(testingPMLinit, 2, function(c) sum(is.na(c)) == nrow(testingPMLinit))`<br />
`testingPMLinit <- testingPMLinit[ , !testingNA]`<br />

# Building and Evaluating a Model

## Consideration

> “Because of the characteristic noise in the sensor data, we used a Random Forest approach. This algorithm is characterized by a subset of features, selected in a random and independent manner with the same distribution for each of the trees in the forest.”<br />

> Velloso, E.; Bulling, A.; Gellersen, H.; Ugulino, W.; Fuks, H. Qualitative Activity Recognition of Weight Lifting Exercises. Proceedings of 4th International Conference in Cooperation with SIGCHI (Augmented Human '13) . Stuttgart, Germany: ACM SIGCHI, 2013.<br />

> Read more: http://groupware.les.inf.puc-rio.br/har#wle_paper_section#ixzz4aNGKR4KC<br />

We will also use Random Forest and see to how well it performs for the purposes of this Assignment.<br />

## Creating Training and Testing Sets from Training Data

With a data set of this size, we can afford more thorough training. We allocate 75% to train the model. The remainder 25% serve to predict the Out of Sample error.<br />

Note: At this step, we factor in the “classe” variable.<br />

`set.seed(12354) # Seed set for reproducibility purposes`<br />
`inTrain <- createDataPartition(y = trainingPMLinit$classe, p = 0.75, list = FALSE)`<br />
`trainingPML <- trainingPMLinit[inTrain, ]`<br />
`testingPML <- trainingPMLinit[-inTrain, ]`<br />

## Training the Model

> Features of Random Forests

> It is unexcelled in accuracy among current algorithms.
> It runs efficiently on large data bases.
> It can handle thousands of input variables without variable deletion.
> It gives estimates of what variables are important in the classification.
> It generates an internal unbiased estimate of the generalization error as the forest building progresses.
> It has an effective method for estimating missing data and maintains accuracy when a large proportion of the data are missing.
> It has methods for balancing error in class population unbalanced data sets.
> Generated forests can be saved for future use on other data.
> Prototypes are computed that give information about the relation between the variables and the classification.
> It computes proximities between pairs of cases that can be used in clustering, locating outliers, or (by scaling) give interesting views of the data.
> The capabilities of the above can be extended to unlabeled data, leading to unsupervised clustering, data views and outlier detection.
> It offers an experimental method for detecting variable interactions.

> Remarks

> Random forests does not overfit. You can run as many trees as you want. It is fast. Running on a data set with 50,000 cases and 100 variables, it produced 100 trees in 11 minutes on a 800Mhz machine. For large data sets the major memory requirement is the storage of the data itself, and three integer arrays with the same dimensions as the data. If proximities are calculated, storage requirements grow as the number of cases times the number of trees.<br />

> (From: Random Forests, Leo Breiman and Adele Cutler)<br />

We will use the caret package’s own model training options for the Random Forest.<br />

Note:<br />

While caret allows for parallel processing and Random Forests are not resource-heavy, 64-bit Windows platforms such as the one used for this Assignment execution on may not handle it too well. Common issues on Windows 10 are outlined here:   https://github.com/tobigithub/R-parallel/wiki/R-parallel-Errors

In comparison, a 64-bit OS-X Macbook Air has no trouble handling the task.<br />

`library(doParallel)`<br />
`cluster <- makeCluster(2)`<br />
`registerDoParallel(cluster)`<br />

`model <- train(classe ~ ., data = trainingPML, method = 'rf', verbose = TRUE)`<br />

`model`<br />
`Random Forest `<br />

`14718 samples`<br />
  ` 52 predictor`<br />
  `  5 classes: 'A', 'B', 'C', 'D', 'E'`<br />

`No pre-processing`<br />
`Resampling: Bootstrapped (25 reps) `<br />
`Summary of sample sizes: 14718, 14718, 14718, 14718, 14718, 14718, ... `<br />
`Resampling results across tuning parameters:`<br />

`  mtry  Accuracy   Kappa    `<br />
`   2    0.9886517  0.9856410`<br />
`  27    0.9894938  0.9867077`<br />
 ` 52    0.9824803  0.9778341`<br />

`Accuracy was used to select the optimal model using  the largest value.`<br />
`The final value used for the model was mtry = 27.`<br />

Best model accuracy as decided by caret (bestTune):<br />

> tune / bestTune This generic function tunes hyperparameters of statistical methods using a grid search over supplied parameter ranges. (R Help pages)<br />

`bestRFmodel <- model$results$Accuracy[as.integer(row.names(model$bestTune))]`

Out of Bag (OOB) error rate of the model based on number of trees used (ntree):<br />

`rateOOBerror <- model$finalModel$err.rate[model$finalModel$ntree,1]`

The results:<br />

`bestRFmodel`
`[1] 0.9894938`

`rateOOBerror`
`        OOB `
`0.005639353`

At ~0.99 accuracy and under ~0.01 error rate, we will be hard-pressed to find a better method.<br />

# Cross Validation

We will apply our chosen model first to the testingPML set which we cut aside earlier from the training data (25%).<br />

`testingPML_classesReference <- testingPML$classe; testingPML$classe <- NULL`

`testingPML_classesPredicted <- predict(model$finalModel, newdata = testingPML)`

`testingPML_confusionMatrix <- confusionMatrix(data = testingPML_classesPredicted, reference = testingPML_classesReference)`

`OutOfSampleErrorRate  <- 1 - testingPML_confusionMatrix$overall['Accuracy']; names(OutOfSampleErrorRate) = 'ErrorRate'`

`testingPML_confusionMatrix`

`Confusion Matrix and Statistics`

`         Reference`
`Prediction    A    B    C    D    E`
 `        A 1393    7    0    0    0`
 `        B    0  940    1    0    0`
 `        C    1    2  852   12    1`
 `        D    0    0    2  792    2`
 `        E    1    0    0    0  898`

`Overall Statistics`
                                         
 `              Accuracy : 0.9941         `
 `                95% CI : (0.9915, 0.996)`
`    No Information Rate : 0.2845         `
`    P-Value [Acc > NIR] : < 2.2e-16      `
                                         
`                  Kappa : 0.9925         `
` Mcnemar's Test P-Value : NA             `

`Statistics by Class:`

`                     Class: A Class: B Class: C Class: D Class: E`
`Sensitivity            0.9986   0.9905   0.9965   0.9851   0.9967`
`Specificity            0.9980   0.9997   0.9960   0.9990   0.9998`
`Pos Pred Value         0.9950   0.9989   0.9816   0.9950   0.9989`
`Neg Pred Value         0.9994   0.9977   0.9993   0.9971   0.9993`
`Prevalence             0.2845   0.1935   0.1743   0.1639   0.1837`
`Detection Rate         0.2841   0.1917   0.1737   0.1615   0.1831`
`Detection Prevalence   0.2855   0.1919   0.1770   0.1623   0.1833`
`Balanced Accuracy      0.9983   0.9951   0.9963   0.9920   0.9982`

We will also apply the same to the original testing data, arranged by problem_id.<br />

`testingPMLinit <- arrange(testingPMLinit, problem_id)`
`testingPMLinit_prediction <- predict(model$finalModel, newdata = testingPMLinit)`

`testingPMLinit_prediction `

` 1  2  3  4  5  6  7  8  9 10 11 12 13 14 15 16 17 18 19 20 `
` B  A  B  A  A  E  D  B  A  A  B  C  B  A  E  E  A  B  B  B `
`Levels: A B C D E`

# Write Result to Files

This will export the initial testing data predictions into indiviual files.<br />

`testingPMLinit_writeFiles = function(x)`
`    {`
`  n = length(x)`
`  for(i in 1:n){`
   ` filename = paste0("problem_id_", i, ".txt")`
  `  write.table(x[i], file=filename, quote=FALSE, row.names=FALSE, col.names=FALSE)}`
  `   }`
`testingPMLinit_writeFiles(testingPMLinit_prediction)`
