library(h2o)

h2o.init()


# library(class)
library(caret)
library(gains)
library(rlang)
library(geiger)
library(data.table)


setwd('/Users/qinqingao/Desktop/Columbia/Courses/Fall 2018/EECS 6690/Project')
# require(gdata)

# raw <- read.xls ("default of credit card clients.xls", sheet = 1, skip = 1, row.names = 1)
# dim(raw)
# # [1] 30000    24

# #check any null value, none (30000 * 24 = 720000)
# table(is.na(raw))
# #  FALSE 
# # 720000 

# #train-test split, 80%-20%
# set.seed(2018)
# sample_row_num <- sample(nrow(raw), nrow(raw) * 0.8)

# train <- raw[sample_row_num, ]
# test <- raw[-sample_row_num, ]  
 
# train_label <- train[, ncol(train)]
# test_label <- test[, ncol(test)]


#export train, test to csv for h2o import
# write.csv(train, 'train.csv', row.names = F)
# write.csv(test, 'test.csv', row.names = F)


# Import train/test set into H2O
train <- h2o.importFile('/Users/qinqingao/Desktop/Columbia/Courses/Fall 2018/EECS 6690/Project/train.csv')
test <- h2o.importFile('/Users/qinqingao/Desktop/Columbia/Courses/Fall 2018/EECS 6690/Project/test.csv')


#normalizing numerical variables
scale01 <- function(x){
     (x - min(x)) / (max(x) - min(x))
 }

train_norm = train
valid_norm = test

for(name in names(train)){
     if(name != "default.payment.next.month"){
         train_norm[name] <- scale01(train_norm[name])
         valid_norm[name] <- scale01(valid_norm[name])
     }     
 }



# Identify response and predictors
y <- "default.payment.next.month"
x <- setdiff(names(train_norm), y)

# For binary classification, response should be a factor
train_norm[,y] <- as.factor(train_norm[,y])
valid_norm[,y] <- as.factor(valid_norm[,y])


#########################

#    5.1  AML Train     # 

#########################


# Run AutoML for 20 base models (limited to 1 hour max runtime by default)
aml <- h2o.automl(x = x, y = y,
                  training_frame = train_norm,
                  max_models = 20,
                  seed = 1)

# View the AutoML Leaderboard
lb <- aml@leaderboard
print(lb, n = nrow(lb))  # Print all rows instead of default (6 rows)
#                                                 model_id       auc
# 1     StackedEnsemble_AllModels_0_AutoML_20181218_122201 0.7832385
# 2              GBM_grid_0_AutoML_20181218_122201_model_1 0.7817236
# 3              GBM_grid_0_AutoML_20181218_122201_model_4 0.7817178
# 4              GBM_grid_0_AutoML_20181218_122201_model_0 0.7814908
# 5  StackedEnsemble_BestOfFamily_0_AutoML_20181218_122201 0.7814570
# 6              GBM_grid_0_AutoML_20181218_122201_model_2 0.7806812
# 7             GBM_grid_0_AutoML_20181218_122201_model_11 0.7763260
# 8              GBM_grid_0_AutoML_20181218_122201_model_9 0.7757523
# 9             GBM_grid_0_AutoML_20181218_122201_model_14 0.7746678
# 10             GBM_grid_0_AutoML_20181218_122201_model_3 0.7740559
# 11             GBM_grid_0_AutoML_20181218_122201_model_7 0.7730768
# 12            GBM_grid_0_AutoML_20181218_122201_model_15 0.7685951
# 13                          XRT_0_AutoML_20181218_122201 0.7628036
# 14             GBM_grid_0_AutoML_20181218_122201_model_5 0.7614312
# 15                 DeepLearning_0_AutoML_20181218_122201 0.7610898
# 16             GBM_grid_0_AutoML_20181218_122201_model_6 0.7607885
# 17                          DRF_0_AutoML_20181218_122201 0.7595923
# 18            GBM_grid_0_AutoML_20181218_122201_model_10 0.7578697
# 19             GBM_grid_0_AutoML_20181218_122201_model_8 0.7563372
# 20            GBM_grid_0_AutoML_20181218_122201_model_13 0.7489941
# 21             GLM_grid_0_AutoML_20181218_122201_model_0 0.7238051
# 22            GBM_grid_0_AutoML_20181218_122201_model_12 0.6690221
#      logloss mean_per_class_error      rmse       mse
# 1  0.4293680            0.2871955 0.3663107 0.1341836
# 2  0.4289458            0.2909941 0.3669946 0.1346850
# 3  0.4278568            0.2926448 0.3662966 0.1341732
# 4  0.4283937            0.2900141 0.3665460 0.1343560
# 5  0.4315509            0.2884573 0.3673627 0.1349554
# 6  0.4294783            0.2937596 0.3672225 0.1348524
# 7  0.4321959            0.2904779 0.3682578 0.1356138
# 8  0.5229811            0.2859855 0.4128099 0.1704120
# 9  0.4425817            0.2890229 0.3721459 0.1384925
# 10 0.4349234            0.2962509 0.3695175 0.1365432
# 11 0.4336698            0.2953465 0.3690750 0.1362164
# 12 0.5225075            0.2921521 0.4126030 0.1702412
# 13 0.4485333            0.2992553 0.3720678 0.1384344
# 14 0.4508919            0.3001577 0.3758290 0.1412475
# 15 0.4425513            0.3016918 0.3721644 0.1385063
# 16 0.5235303            0.2958110 0.4130406 0.1706025
# 17 0.4512426            0.3076536 0.3734098 0.1394348
# 18 0.4587117            0.3139602 0.3771194 0.1422191
# 19 0.4635691            0.3099325 0.3801896 0.1445441
# 20 0.4815355            0.3115117 0.3851673 0.1483538
# 21 0.4656582            0.3158674 0.3808495 0.1450464
# 22 1.7856000            0.3667446 0.4704448 0.2213183

# [22 rows x 6 columns] 


# The leader model is stored here
aml@leader

# Model Details:
# ==============

# H2OBinomialModel: stackedensemble
# Model ID:  StackedEnsemble_AllModels_0_AutoML_20181218_122201 
# NULL


# H2OBinomialMetrics: stackedensemble
# ** Reported on training data. **

# MSE:  0.1029609
# RMSE:  0.3208752
# LogLoss:  0.343227
# Mean Per-Class Error:  0.176618
# AUC:  0.9153937
# Gini:  0.8307875

# Confusion Matrix (vertical: actual; across: predicted) for F1-optimal threshold:
#            0    1    Error         Rate
# 0      12722 2206 0.147776  =2206/14928
# 1        873 3376 0.205460    =873/4249
# Totals 13595 5582 0.160557  =3079/19177

# Maximum Metrics: Maximum metrics at their respective thresholds
#                         metric threshold    value idx
# 1                       max f1  0.199586 0.686807 272
# 2                       max f2  0.160956 0.793377 302
# 3                 max f0point5  0.555479 0.710136 125
# 4                 max accuracy  0.416895 0.860562 171
# 5                max precision  0.909702 1.000000   0
# 6                   max recall  0.089505 1.000000 383
# 7              max specificity  0.909702 1.000000   0
# 8             max absolute_mcc  0.185125 0.593517 281
# 9   max min_per_class_accuracy  0.185125 0.830252 281
# 10 max mean_per_class_accuracy  0.160956 0.837867 302

# Gains/Lift Table: Extract with `h2o.gainsLift(<model>, <data>)` or `h2o.gainsLift(<model>, valid=<T/F>, xval=<T/F>)`
# H2OBinomialMetrics: stackedensemble
# ** Reported on validation data. **

# MSE:  0.1365333
# RMSE:  0.3695042
# LogLoss:  0.4341799
# Mean Per-Class Error:  0.2849271
# AUC:  0.78472
# Gini:  0.56944

# Confusion Matrix (vertical: actual; across: predicted) for F1-optimal threshold:
#           0    1    Error        Rate
# 0      3098  655 0.174527   =655/3753
# 1       423  647 0.395327   =423/1070
# Totals 3521 1302 0.223512  =1078/4823

# Maximum Metrics: Maximum metrics at their respective thresholds
#                         metric threshold    value idx
# 1                       max f1  0.207969 0.545531 257
# 2                       max f2  0.118178 0.642196 343
# 3                 max f0point5  0.456777 0.578167 154
# 4                 max accuracy  0.456777 0.819200 154
# 5                max precision  0.898803 1.000000   0
# 6                   max recall  0.077598 1.000000 397
# 7              max specificity  0.898803 1.000000   0
# 8             max absolute_mcc  0.306230 0.419267 208
# 9   max min_per_class_accuracy  0.160829 0.714019 294
# 10 max mean_per_class_accuracy  0.160829 0.715256 294

# Gains/Lift Table: Extract with `h2o.gainsLift(<model>, <data>)` or `h2o.gainsLift(<model>, valid=<T/F>, xval=<T/F>)`
# H2OBinomialMetrics: stackedensemble
# ** Reported on cross-validation data. **
# ** 5-fold cross-validation on training data (Metrics computed for combined holdout predictions) **

# MSE:  0.1341836
# RMSE:  0.3663107
# LogLoss:  0.429368
# Mean Per-Class Error:  0.2871955
# AUC:  0.7832385
# Gini:  0.5664771

# Confusion Matrix (vertical: actual; across: predicted) for F1-optimal threshold:
#            0    1    Error         Rate
# 0      12565 2363 0.158293  =2363/14928
# 1       1768 2481 0.416098   =1768/4249
# Totals 14333 4844 0.215414  =4131/19177

# Maximum Metrics: Maximum metrics at their respective thresholds
#                         metric threshold    value idx
# 1                       max f1  0.222920 0.545694 255
# 2                       max f2  0.126651 0.639719 337
# 3                 max f0point5  0.474933 0.587540 149
# 4                 max accuracy  0.474933 0.822913 149
# 5                max precision  0.910431 1.000000   0
# 6                   max recall  0.076970 1.000000 397
# 7              max specificity  0.910431 1.000000   0
# 8             max absolute_mcc  0.372126 0.422820 187
# 9   max min_per_class_accuracy  0.159115 0.706257 303
# 10 max mean_per_class_accuracy  0.197921 0.715047 272

# Gains/Lift Table: Extract with `h2o.gainsLift(<model>, <data>)` or `h2o.gainsLift(<model>, valid=<T/F>, xval=<T/F>)`


# pred <- h2o.predict(aml, train_norm)  # predict(aml, test) also works

# or:
pred <- h2o.predict(aml@leader, train_norm)
#   predict        p0        p1
# 1       0 0.8061128 0.1938872
# 2       1 0.7753841 0.2246159
# 3       1 0.4764972 0.5235028
# 4       1 0.5950168 0.4049832
# 5       1 0.6069764 0.3930236
# 6       1 0.7753841 0.2246159

# [6000 rows x 3 columns] 


#find range of p1 given predict == 1
range(as.data.frame(pred)[which(as.data.frame(pred)[,1] == 1), ][,3])
# [1] 0.2283783 0.9107308

#confusion matrix for train
PredAMLLabel = ifelse(pred[,3] > 0.5, 1, 0)
err <- 1 - confusionMatrix(table(as.vector(PredAMLLabel), as.vector(train_norm[,y])))$overall['Accuracy']
err
# 0.176375

# Confusion Matrix and Statistics

   
#         0     1
#   0 18054  2991
#   1   627  2328
                                          
#                Accuracy : 0.8492          
#                  95% CI : (0.8447, 0.8538)
#     No Information Rate : 0.7784          
#     P-Value [Acc > NIR] : < 2.2e-16       
                                          
#                   Kappa : 0.4805          
#  Mcnemar's Test P-Value : < 2.2e-16       
                                          
#             Sensitivity : 0.9664          
#             Specificity : 0.4377          
#          Pos Pred Value : 0.8579          
#          Neg Pred Value : 0.7878          
#              Prevalence : 0.7784          
#          Detection Rate : 0.7522          
#    Detection Prevalence : 0.8769          
#       Balanced Accuracy : 0.7021          
                                          
#        'Positive' Class : 0    



PredAMLLabel = data.frame(as.numeric(as.vector(PredAMLLabel)))
names(PredAMLLabel) <- "PredAMLLabel"

PredAMLScore = as.vector(pred[,3])
length(PredAMLScore)

PredAMLScore = data.frame(as.vector(PredAMLScore))
names(PredAMLScore) <- "PredAMLScore"


#normalizing numerical variables
scale01 <- function(x){
     (x - min(x)) / (max(x) - min(x))
 }

train_norm = train
valid_norm = test

for(name in names(train)){
     if(name != "default.payment.next.month"){
         train_norm[name] <- scale01(train_norm[name])
         valid_norm[name] <- scale01(valid_norm[name])
     }     
 }


train_norm = data.frame(as.data.frame(train_norm), PredAMLLabel, PredAMLScore)
head(train_norm)

gtt = gains(actual=as.numeric(train_norm[,y]), predicted=train_norm$PredAMLScore,optimal=TRUE)
cpt_y = gtt$cume.pct.of.total
cpt_x = gtt$depth

predicted = table(train_norm$PredAMLLabel)[2]
predicted

xx = cpt_x / 100 * 24000
yy = cpt_y * predicted

xx = prepend(xx,0,before=1)
yy = prepend(yy,0,before=1)
fit = lm(yy ~ poly(xx,3,raw=TRUE))

xx = 0:24000
model_yy = predict(fit,data.frame(xx))


# png("/Users/qinqingao/Desktop/Columbia/Courses/Fall 2018/EECS 6690/Project/figs/AML_gain_chart_train.png")
plot(xx, model_yy, col="green",xlab="Number of total data", ylab="Cumulative number of target data",
	type = 'l', lwd = 3)

best_yy = rep(predicted, 24001)
for(i in 0:predicted){
  best_yy[i+1] = i
}

lines(xx,best_yy,col="red",lwd=2)

base_yy = predicted / 24000 * xx
lines(xx,base_yy,col="blue", lwd = 2)

legend('bottomright', legend=c("best curve","model", "baseline"), col = c("red","green","blue"), lwd=c(1,1,1),cex=1)
title("Gain chart of AML (training)")
# dev.off()


#Calculate area ratio
a1v = sum(model_yy-base_yy)
a2v = sum(best_yy-base_yy)
a1v/a2v
# [1] 0.6851915


#########################

#    5.2  AML Test     # 

#########################


predv <- h2o.predict(aml@leader, valid_norm)

#find range of p1 given predict == 1
range(as.data.frame(pred)[which(as.data.frame(predv)[,1] == 1), ][,3])
# [1] 0.2283783 0.9107308

#confusion matrix for test
PredAMLLabelV = ifelse(predv[,3] > 0.5, 1, 0)
errv <- 1 - confusionMatrix(table(as.vector(PredAMLLabelV), as.vector(valid_norm[,y])))$overall['Accuracy']
errv
# 0.1795


PredAMLLabelV = data.frame(as.numeric(as.vector(PredAMLLabelV)))
names(PredAMLLabelV) <- "PredAMLLabelV"

PredAMLScoreV = as.vector(predv[,3])
length(PredAMLScoreV)

PredAMLScoreV = data.frame(as.vector(PredAMLScoreV))
names(PredAMLScoreV) <- "PredAMLScoreV"


# #normalizing numerical variables
# scale01 <- function(x){
#      (x - min(x)) / (max(x) - min(x))
#  }

# train_norm = train
# valid_norm = test

# for(name in names(train)){
#      if(name != "default.payment.next.month"){
#          train_norm[name] <- scale01(train_norm[name])
#          valid_norm[name] <- scale01(valid_norm[name])
#      }     
#  }


valid_norm = data.frame(as.data.frame(valid_norm), PredAMLLabelV, PredAMLScoreV)
head(valid_norm)

gtt = gains(actual=as.numeric(valid_norm[,y]), predicted=valid_norm$PredAMLScoreV,optimal=TRUE)
cpt_y = gtt$cume.pct.of.total
cpt_x = gtt$depth

predictedV = table(valid_norm$PredAMLLabelV)[2]
predictedV

xx = cpt_x / 100 * 6000
yy = cpt_y * predictedV

xx = prepend(xx,0,before=1)
yy = prepend(yy,0,before=1)
fit = lm(yy ~ poly(xx,3,raw=TRUE))

xx = 0:6000
model_yy = predict(fit,data.frame(xx))


# png("/Users/qinqingao/Desktop/Columbia/Courses/Fall 2018/EECS 6690/Project/figs/AML_gain_chart_test.png")
plot(xx, model_yy, col="green",xlab="Number of total data", ylab="Cumulative number of target data",
	type = 'l', lwd = 3)

best_yy = rep(predictedV, 6001)
for(i in 0:predictedV){
  best_yy[i+1] = i
}

lines(xx,best_yy,col="red",lwd=2)

base_yy = predictedV / 6000 * xx
lines(xx,base_yy,col="blue", lwd = 2)

legend('bottomright', legend=c("best curve","model", "baseline"), col = c("red","green","blue"), lwd=c(1,1,1),cex=1)
title("Gain chart of AML (validation)")
# dev.off()


#Calculate area ratio
a1v = sum(model_yy-base_yy)
a2v = sum(best_yy-base_yy)
a1v/a2v
# [1] 0.4645352


#Part 5.2.3: Use SSM(Sorting Smoothing Method) to estimate real probability
#1. order the valid data according to predictive probability
valid_norm_sort = valid_norm[order(PredAMLScoreV), ]
#2. use SSM formula to evaluate actural probability 'Pi', we choose n =50 according to the paper
VALIDSIZE = dim(valid_norm)[1]

n = 50
actural_p_valid = rep(0, VALIDSIZE)
pred_valid = round(valid_norm_sort$PredAMLScoreV)
pred_valid = prepend(pred_valid, rep(0, n), before=1)
pred_valid = append(pred_valid, rep(0, n))

for(i in 1:VALIDSIZE){
  actural_p_valid[i] = sum(pred_valid[i:(i + 2 * n)])/(2 * n + 1)
}
valid_norm_sort = data.frame(valid_norm_sort, actural_p_valid)


# png("/Users/qinqingao/Desktop/Columbia/Courses/Fall 2018/EECS 6690/Project/figs/Scatter plot diagram of AML.png")
plot(valid_norm_sort$PredAMLScoreV, valid_norm_sort$actural_p_valid,
	xlab = "Predicted Probability", ylab = "Actual probability")

yy = valid_norm_sort$actural_p_valid
xx = valid_norm_sort$PredAMLScoreV
actual_fit = lm(yy ~ xx)

xx = seq(0, 1 : 0.1)
yy = predict(actual_fit, data.frame((xx)))
lines(xx, yy)

summary(actual_fit)
legend('topleft',legend=c("y = 1.542782x - 0.252073","R^2 = 0.827"))
title('Actual vs Pred Probability of AML (validation)')
# dev.off()














