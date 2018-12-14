## EECS 6690 Project
## Credit Card


#import data
setwd('/Users/qinqingao/Desktop/Columbia/Courses/Fall 2018/EECS 6690/Project')
require(gdata)
raw <- read.xls ("default of credit card clients.xls", sheet = 1, skip = 1, row.names = 1)
dim(raw)
# [1] 30000    24


#train-test split, 80%-20%
set.seed(2018)
sample_row_num <- sample(nrow(raw), nrow(raw) * 0.8)

train <- raw[sample_row_num, ]
test <- raw[-sample_row_num, ]  

train_label <- train[, ncol(train)]
test_label <- test[, ncol(test)]


###### knn ###### 

#train data with knn
library(class)


#train error
knn_35_train <- knn(train = train, test = train, cl = train_label, k = 35)
ACC_35_train <- sum(train_label == knn_35_train)/NROW(train_label)
Err_train <- 1 - ACC_35_train
Err_train
# [1] 0.215875


#test error
knn_35_test <- knn(train = train, test = test, cl = train_label, k = 35)
ACC_35_test <- sum(test_label == knn_35_test)/NROW(test_label)
Err_test <- 1 - ACC_35_test
Err_test
# [1] 0.2156667


library(caret)
confusionMatrix(table(knn_35_train, train_label))
confusionMatrix(table(knn_35_test, test_label))


#select optimal k
# i = 1
# k_optm = 1                     
# for (i in 1 : 60){ 
#     knn_mod <-  knn(train = train, test = test, cl = train_label, k = i)
#     k_optm[i] <- 100 * sum(test_label == knn_mod)/NROW(test_label)
#     # k = i
#     # cat(k,'=', k_optm[i],'\n')       # to print % accuracy 
# }

# plot(k_optm, type = "b", xlab = "K-Value", ylab = "Accuracy level")


library(gains)
library(rlang)
library(geiger)

#plot train lift curve for kNN
gknn_t = gains(actual = train_label, predicted = as.numeric(knn_35_train), groups = 2, optimal = TRUE)
cpt_y_t = gknn_t$cume.pct.of.total
cpt_y_t
# [1] 0.09494266 1.00000000

cpt_x_t = gknn_t$depth
cpt_x_t
# [1]   4 100


predicted_d_t = table(knn_35_train)[2]
xx_t = cpt_x_t / 100 * 24000
yy_t = cpt_y_t * as.numeric(predicted_d_t)
plot(xx_t, yy_t)

xx_t = prepend(xx_t, 0, before = 1)
yy_t = prepend(yy_t, 0, before = 1)
fit_t = lm(yy_t ~ poly(xx_t, 3, raw = TRUE))

xx_t = 0 : 24000
model_yy_t = predict(fit_t, data.frame(xx_t))


png("/Users/qinqingao/Desktop/Columbia/Courses/Fall 2018/EECS 6690/Project/figs/kNN_lift_chart_train.png")
plot(xx_t, model_yy_t, col = "green", xlab = "Number of total data", ylab = "Cumulative number of target data", type = 'l', lwd = 3, ylim = c(0, 1000))

best_yy_t = rep(predicted_d_t, 24001)
for(i in 0 : 24000){
	if (i <= predicted_d_t){
		best_yy_t[i] = i
						   } 
	else{
  best_yy_t[i + 1] = predicted_d_t
		}
				    }
lines(xx_t, best_yy_t, col = "red", lwd = 2)

base_yy_t = (predicted_d_t / 24000) * xx_t
lines(xx_t, base_yy_t, col = "blue", lwd = 2)
grid(nx = NA, ny = NULL)

legend('bottomright', legend = c("best curve", "model", "baseline"), col = c("red","green","blue"), 
	cex = 1, lwd = 2)
title("Lift chart of kNN (training)")
dev.off()


# calculate area ratio
a1_t = sum(model_yy_t - base_yy_t)
a2_t = sum(best_yy_t - base_yy_t)
area_ratio_t <- a1_t/a2_t
area_ratio_t
# [1] 0.4948719



#plot test lift curve for kNN
gknn_v = gains(actual = test_label, predicted = as.numeric(knn_35_test), groups = 2, optimal = TRUE)
cpt_y_v = gknn_v$cume.pct.of.total
cpt_y_v
# [1] 0.09567198 1.00000000

cpt_x_v = gknn_v$depth
cpt_x_v
# [1]   4 100


predicted_d_v = table(knn_35_test)[2]
xx_v = cpt_x_v / 100 * 6000
yy_v = cpt_y_v * as.numeric(predicted_d_v)
plot(xx_v, yy_v)

xx_v = prepend(xx_v, 0, before = 1)
yy_v = prepend(yy_v, 0, before = 1)
fit_v = lm(yy_v ~ poly(xx_v, 3, raw = TRUE))

xx_v = 0 : 6000
model_yy_v = predict(fit_v, data.frame(xx_v))


png("/Users/qinqingao/Desktop/Columbia/Courses/Fall 2018/EECS 6690/Project/figs/kNN_lift_chart_test.png")
plot(xx_v, model_yy_v, col = "green", xlab = "Number of total data", ylab = "Cumulative number of target data", 
	type = 'l', lwd = 3, ylim = c(0, 250))

best_yy_v = rep(predicted_d_v, 6001)
for(i in 0 : 6000){
	if (i <= predicted_d_v){
		best_yy_v[i] = i
						   } 
	else{
  best_yy_v[i + 1] = predicted_d_v
		}
				    }
lines(xx_v, best_yy_v, col = "red", lwd = 2)

base_yy_v = (predicted_d_v / 6000) * xx_v
lines(xx_v, base_yy_v, col = "blue", lwd = 2)
grid(nx = NA, ny = NULL)

legend('bottomright', legend = c("best curve", "model", "baseline"), col = c("red","green","blue"), 
	cex = 1, lwd = 2)
title("Lift chart of kNN (validation)")
dev.off()


# calculate area ratio
a1_v = sum(model_yy_v - base_yy_v)
a2_v = sum(best_yy_v - base_yy_v)
area_ratio_v <- a1_v/a2_v
area_ratio_v
# [1] 0.5022661



#################################

#      logistic regression      # 

#################################


lr <- glm(default.payment.next.month ~ . , family = binomial(link = 'logit'), data = train)
# summary(lr)
# anova(lr, test = "Chisq")


#train error
train_results <- predict(lr, newdata = train, type = 'response')
train_results <- ifelse(train_results > 0.5, 1, 0)
train_misClasificError <- mean(train_results != train$default.payment.next.month)
train_misClasificError
# [1] 0.1895833


#test error
test_results <- predict(lr, newdata = test, type = 'response')
test_results <- ifelse(test_results > 0.5, 1, 0)
misClasificError <- mean(test_results != test$default.payment.next.month)
misClasificError
# [1] 0.1863333



#plot train lift curve for LR
glr_t = gains(actual = train_label, predicted = as.numeric(train_results), groups = 2, optimal = TRUE)
cpt_y_lr_t = glr_t$cume.pct.of.total
cpt_y_lr_t
# [1] 0.2404587 1.0000000

cpt_x_lr_t = glr_t$depth
cpt_x_lr_t
# [1]   7 100


predicted_d_lr_t = table(train_results)[2]
predicted_d_lr_t
#    1 
# 1789
xx_lr_t = cpt_x_lr_t / 100 * 24000
yy_lr_t = cpt_y_lr_t * as.numeric(predicted_d_lr_t)
plot(xx_lr_t, yy_lr_t)

xx_lr_t = prepend(xx_lr_t, 0, before = 1)
yy_lr_t = prepend(yy_lr_t, 0, before = 1)
fit_lr_t = lm(yy_lr_t ~ poly(xx_lr_t, 3, raw = TRUE))

xx_lr_t = 0 : 24000
model_yy_lr_t = predict(fit_lr_t, data.frame(xx_lr_t))


png("/Users/qinqingao/Desktop/Columbia/Courses/Fall 2018/EECS 6690/Project/figs/lr_lift_chart_train.png")
plot(xx_lr_t, model_yy_lr_t, col = "green", xlab = "Number of total data", ylab = "Cumulative number of target data", 
	type = 'l', lwd = 3, ylim = c(0, 3000))

best_yy_lr_t = rep(predicted_d_lr_t, 24001)
for(i in 0 : 24000){
	if (i <= predicted_d_lr_t){
		best_yy_lr_t[i] = i
						   } 
	else{
  best_yy_lr_t[i + 1] = predicted_d_lr_t
		}
				    }
lines(xx_lr_t, best_yy_lr_t, col = "red", lwd = 2)

base_yy_lr_t = (predicted_d_lr_t / 24000) * xx_lr_t
lines(xx_lr_t, base_yy_lr_t, col = "blue", lwd = 2)
grid(nx = NA, ny = NULL)

legend('bottomright', legend = c("best curve", "model", "baseline"), col = c("red","green","blue"), 
	cex = 1, lwd = 2)
title("Lift chart of Logistic Regression (training)")
dev.off()


# calculate area ratio
a1_lr_t = sum(model_yy_lr_t - base_yy_lr_t)
a2_lr_t = sum(best_yy_lr_t - base_yy_lr_t)
area_ratio_lr_t <- a1_lr_t/a2_lr_t
area_ratio_lr_t
# [1] 0.9430203



#plot test lift curve for lg
glr_v = gains(actual = test_label, predicted = as.numeric(test_results), groups = 2, optimal = TRUE)
cpt_y_lr_v = glr_v$cume.pct.of.total
cpt_y_lr_v
# [1] 0.2437358 1.00000000

cpt_x_lr_v = glr_v$depth
cpt_x_lr_v
# [1]   7 100


predicted_d_lr_v = table(test_results)[2]
predicted_d_lr_v
#  1 
# 443
xx_lr_v = cpt_x_lr_v / 100 * 6000
yy_lr_v = cpt_y_lr_v * as.numeric(predicted_d_lr_v)
plot(xx_lr_v, yy_lr_v)

xx_lr_v = prepend(xx_lr_v, 0, before = 1)
yy_lr_v = prepend(yy_lr_v, 0, before = 1)
fit_lr_v = lm(yy_lr_v ~ poly(xx_lr_v, 3, raw = TRUE))

xx_lr_v = 0 : 6000
model_yy_lr_v = predict(fit_lr_v, data.frame(xx_lr_v))


png("/Users/qinqingao/Desktop/Columbia/Courses/Fall 2018/EECS 6690/Project/figs/lr_lift_chart_test.png")
plot(xx_lr_v, model_yy_lr_v, col = "green", xlab = "Number of total data", ylab = "Cumulative number of target data", 
	type = 'l', lwd = 3, ylim = c(0, 600))

best_yy_lr_v = rep(predicted_d_lr_v, 6001)
for(i in 0 : 6000){
	if (i <= predicted_d_lr_v){
		best_yy_lr_v[i] = i
						   } 
	else{
  best_yy_lr_v[i + 1] = predicted_d_lr_v
		}
				    }
lines(xx_lr_v, best_yy_lr_v, col = "red", lwd = 2)

base_yy_lr_v = (predicted_d_lr_v / 6000) * xx_lr_v
lines(xx_lr_v, base_yy_lr_v, col = "blue", lwd = 2)
grid(nx = NA, ny = NULL)

legend('bottomright', legend = c("best curve", "model", "baseline"), col = c("red","green","blue"), 
	cex = 1, lwd = 2)
title("Lift chart of Logistic Regression (validation)")
dev.off()


# calculate area ratio
a1_lr_v = sum(model_yy_lr_v - base_yy_lr_v)
a2_lr_v = sum(best_yy_lr_v - base_yy_lr_v)
area_ratio_lr_v <- a1_lr_v/a2_lr_v
area_ratio_lr_v
# [1] 0.9601554







# #plot lift chart
# tot <- 0:15000
# targ <- 0.2212 * tot
# plot(tot, targ, type = 'l', lwd = 2, xlab = "Number of total data", ylab = "Cumulative number of target data", yaxt = "n", ylim = c(0, 3500))
# axis(2, at = seq(0, 3500, by = 500), las = 2)


# segments(0, 0, 0.2212 * 15000, 0.2212 * 15000, lwd = 3)
# segments(0.2212 * 15000, 0.2212 * 15000, 15000, 0.2212 * 15000, lwd = 3)

# grid(nx = NA, ny = NULL)







