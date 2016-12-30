# Build on Chippy's script which score 1128 on public leaderboard
# https://www.kaggle.com/nigelcarpenter/allstate-claims-severity/farons-xgb-starter-ported-to-r
# This great script developed by my teammate Ayush scored 1112 on public leaderboard
# I amend and annotate this script to learn the method.

library(data.table) # for "data table" manipulation, a kind of upgraded data frame
library(xgboost) # build gradient boost tree model 
library(Metrics) # matrix manipulation
library(foreach) # simplify for-loop

#-------- Prepare Data ----------#
ID = 'id'
TARGET = 'loss'
SEED = 0
SHIFT = 200
TRAIN_FILE = "allstate_train.csv"
TEST_FILE = "allstate_test.csv"

# Import data
train = fread(TRAIN_FILE, showProgress = TRUE)
test = fread(TEST_FILE, showProgress = TRUE)

# Shift log for better normality of dependent variable "loss"
y_train = log(train[,TARGET, with = FALSE] + SHIFT)[[TARGET]]

# Remove "id" &"loss". Since the data need to convert to matrix for xgboost modeling.
train[, c(ID, TARGET) := NULL] 
test[, c(ID) := NULL]
# train=train[, c(ID, TARGET):= NULL] may avoid unnecessary r console output.

# Combind train data and test data for one-hot encoding processing.
testframe <- as.data.frame(test) # cast data table object to data frame object
trainframe <- as.data.frame(train)
# I think the cast action is unnecessary...
colnames(trainframe)
colnames(testframe)
ntrain = nrow(train)
train_test = rbind(train, test)

#---------- One-Hot Encoding -----------#
features = names(train_test)

for (f in features) {
  if (class(train_test[[f]])=="character") {
    levels <- sort(unique(train_test[[f]]))
    train_test[[f]] <- as.integer(factor(train_test[[f]], levels=levels))
  }
}
# Some suggests model.matrix or FeatureHashing::hashed.model.matrix can fullfil the encoding work.
# Use model.matrix will have memory error. Not sure about hased.model.matrix.

# Split data
x_train = train_test[1:ntrain,]
x_test = train_test[(ntrain+1):nrow(train_test),]

# Cast data frame object to xgboost matrix object
dtrain = xgb.DMatrix(as.matrix(x_train), label=y_train)
dtest = xgb.DMatrix(as.matrix(x_test))

#---------- Build xgboost model -----------#
set.seed(123)

xgb_params = list(
  seed = 0,
  colsample_bytree = 0.5, # sampling ratio of input variables
  subsample = 0.8, # ratio of training instance
  eta = 0.01, # learning rate, slow but robust
  objective = 'reg:linear',
  max_depth = 12, 
  alpha = 1, # regularization term on weights
  gamma = 2, # minimun loss reduction for a node partition
  min_child_weight = 1,
  base_score = 7.76 # initial prediction score of all instances
)

# Dependent variable y is shift logged, therefore we need to get exponentializing y 
# and predict values first, and then calculate mae
xg_eval_mae <- function (yhat, dtrain) {
  y = getinfo(dtrain, "label")
  err= mae(exp(y),exp(yhat) )
  return (list(metric = "error", value = err))
}

res = xgb.cv(xgb_params,
             dtrain,
             nrounds=5000, 
             nfold=5,
             early_stopping_rounds=15,
             print_every_n = 10,
             verbose= 2, # print performance
             feval=xg_eval_mae,
             maximize=FALSE)
res$train.error.mean
res$test.error.mean
res
#best_nrounds = res$best_iteration # for xgboost v0.6 users 
best_nrounds = which.min(res[, res$train.error.mean]) # for xgboost v0.4-4 users

#id <- which(res[, res$test.error.mean]==best_nrounds)
cv_mean = res$test.error.mean[best_nrounds]
cv_std = res$test.error.std[best_nrounds]

cat(paste0('CV-Mean: ',cv_mean,' ', cv_std))
#best_nrounds 3410
#CV-Mean: 1134.61902176964 5.41080688478779


#------------ Ensemble xgboost models ------------#

set.seed(123)

iterations <- 5
best_nrounds <- c(4995,4996,4997,4998,4999)
best_nrounds <- as.array(best_nrounds)
best_nrounds[1]

#For submission - test data
predictions <- foreach(m=1:iterations,.combine=cbind) %do% { # use foreach to simplify for-loop
  gbdt = xgb.train(xgb_params, dtrain, nrounds = as.integer(best_nrounds[m]/0.8),verbose= 2 )
  # train 5 xgboost tree models.
  exp(predict(gbdt,dtest)) - SHIFT
} 

predictions<- rowMeans(predictions) # averaging 5 columns of prediction values
loss_pred_ensemble <- as.data.frame(predictions)
write.csv(loss_pred_ensemble,'xgb5_ensemble_loss.csv',row.names = FALSE)

#For MAE - Train data
predictions_mae <- foreach(m=1:iterations,.combine=cbind) %do% {
  gbdt = xgb.train(xgb_params, dtrain, nrounds = as.integer(best_nrounds[m]/0.8),verbose= 2 )
  exp(predict(gbdt,dtrain)) - SHIFT
}

predmae_original <- predictions_mae
predictions_mae <- rowMeans(predictions_mae)
predictions_mae <- as.data.frame(predictions_mae)
head(predictions_mae)
write.csv(predictions_mae,'xgb3_ensemble_train_loss.csv',row.names = FALSE)
write.csv(predmae_original,'xgb3_all_train_loss.csv',row.names = FALSE)
mae_train <- mean(abs(predictions_mae-trainframe$loss))
mae_train


#---------- ENSEMBLE XGBOOST+NN ----------#
# Ensemble of ensembles, read in 2 ensembles first
xgb_pred_test <- read.csv("xgb3_ensemble_loss.csv")
xgb_pred_test = as.data.frame(xgb_pred_test)
head(xgb_pred_test)
nn_pred_test = read.csv("allstate-nn-submission1.csv")
nn_pred_test = as.data.frame(nn_pred_test)
head(nn_pred_test)
id <- as.data.frame(testframe$id)
head(id)
colnames(id) = "id"
nn_pred_test$id <- NULL
ensemble_xgb_NN_submission <- (nn_pred_test+xgb_pred_test*9)/10
head(ensemble_xgb_NN_submission)
ensemble_xgb_NN_submission <- as.data.frame(ensemble_xgb_NN_submission)
ensemble_xgb_NN_submission <- as.data.frame(c(id,ensemble_xgb_NN_submission))
head(ensemble_xgb_NN_submission)
write.csv(ensemble_xgb_NN_submission,'ensemble_xgb_NN_submission.csv',row.names = FALSE)

# Combining xgboost models and h2o deeplearning models together didn't yield better prediction
# than xgboost ensembles. Tried to adjust weights, but no improvement.
