######
#Ioannis Skiadas
#Heart Failure Prediction Project
#05Dec2023
###
#INTRODUCTION
#The data set comprises eleven features of patients, with or without a Heart Failure condition. The aim of the project is to assess different algorithms /n
#to accurately classify the morbidity. Such systems may be instrumental to the prevention of severe disease.
#Thus the Heart Failure featue will be the dependent variable. 

#install needed packages
if(!require(tidyverse)) install.packages("tidyverse", repos = "http://cran.us.r-project.org")
if(!require(caret)) install.packages("caret", repos =  "http://cran.us.r-project.org")
if(!require(GGally)) install.packages("GGally", repos= "http://cran.us.r-project.org")
if(!require(xgboost)) install.packages("xgboost", repos ="http://cran.us.r-project.org")
if(!require(knitr)) install.packages("knitr", repos="http://cran.us.r-project.org")
if(!require(nnet)) install.packages("nnet", repos = "http://cran.us.r-project.org")
if(!require(randomForest)) install.packages("randomForest", repos="http://cran.us.r-project.org")

library(readr)
library(tidyverse)
library(caret)
library(GGally)
library(xgboost)
library(knitr)
library(nnet)
library(randomForest)

#load data set
heartDataSet<-read.csv("heart.csv")
summary(heartDataSet)


#present a description of the features


HDS<-heartDataSet%>%mutate(across(where(is.character), as.factor))
summary(HDS)

attr_heart<-c("Age: age of the patient [years]",
              " Sex: sex of the patient [M: Male, F: Female]",
              "ChestPainType: chest pain type [TA: Typical Angina, ATA: Atypical Angina, NAP: Non-Anginal Pain, ASY: Asymptomatic]",
              "RestingBP: resting blood pressure [mm Hg]",
                            
              "Cholesterol: serum cholesterol [mm/dl]", "FastingBS: fasting blood sugar [1: if FastingBS > 120 mg/dl, 0: otherwise]",
              "RestingECG: resting electrocardiogram results [Normal: Normal, ST: having ST-T wave abnormality (T wave inversions and/or ST elevation or depression of > 0.05 mV), LVH: showing probable or definite left ventricular hypertrophy by Estes' criteria]",
              "MaxHR: maximum heart rate achieved [Numeric value between 60 and 202]",
              "ExerciseAngina: exercise-induced angina [Y: Yes, N: No]",
              "Oldpeak: oldpeak = ST [Numeric value measured in depression]",
              "ST_Slope: the slope of the peak exercise ST segment [Up: upsloping, Flat: flat, Down: downsloping]",
              "HeartDisease: output class [1: heart disease, 0: Normal]")
df_attr_heart<-data.frame(index=(1:12), Attributes=attr_heart)
kable(df_attr_heart, format="markdown", caption="Heart Failure Dataset Attributes")

#correlations among the numeric variables are week, thus it is not likely that they will be multicollinearity issues in our models
ggpairs(heartDataSet, columns =c(1,4,5,8,10))

#create train and test

set.seed(1967, sample.kind="Rounding")
HDS_index<-createDataPartition(HDS$HeartDisease, times=1, p=0.25, list = FALSE)
HDS_train<-HDS[-HDS_index,]
HDS_test<-HDS[HDS_index,]






#First a model based on the k-nearest neighbors approach. 
#train the model:
train_knn<-train(as.factor(HeartDisease) ~ ., data=HDS_train, method="knn", tuneGrid=data.frame(k=seq(5,75,5)))
ggplot(train_knn, highlight=TRUE)
#the best k,  
knn_bestTune<-train_knn$bestTune


knn_bestTune$k
#Based on the above, the classification will be:
train_knn_best<-knn3(as.factor(HeartDisease) ~ . , data=HDS_train, k=40)
y_train_knn<-predict(train_knn_best, HDS_test, type="class")

#And the accuracy would be determined as:

cm_knn<-confusionMatrix(y_train_knn, as.factor(HDS_test$HeartDisease), positive = "1")

Results<-tibble(Model="knn", Accuracy=cm_knn$overall["Accuracy"])
kable(Results, format="pipe", caption="Model Accuracy")

#Logistic Regression could also be used:
#GLM
glm_fit<-glm(HeartDisease ~ ., data=HDS_train, family="binomial")

prediction<-predict(glm_fit, newdata = HDS_test, type="response")
logReg<-ifelse(prediction>=0.5, 1, 0)%>%factor()

cm_glm<-confusionMatrix(logReg, as.factor(HDS_test$HeartDisease))

Results<-add_row(Results, Model="Logistic Regression", Accuracy=cm_glm$overall["Accuracy"])
kable(Results, format="pipe", caption="Model Accuracy")

#eXtreme Gradient Boosting ensembles have often been used for classification problems successfully.

#The dimentions of the train and test sets
dim(HDS_train)[1]
dim(HDS_test)[1]

#Prepare the train, test and validation sets.
set.seed(1967, sample.kind ="Rounding")
train_xgb_index<-createDataPartition(HDS_train$HeartDisease, times=1, p=0.25, list=FALSE)
train_xgb<-HDS_train[-train_xgb_index,]
train_xgb_val<-HDS_train[train_xgb_index, -12]
train_xgb<-HDS_train[-train_xgb_index, -12]

train_xgb_val_lab<-HDS_train[train_xgb_index, 12]
train_xgb_lab<-HDS_train[-train_xgb_index, 12]
test_xgb<-HDS_test[,-12]
test_lab<-HDS_test[,12]


xgbtrain<-xgb.DMatrix(data=data.matrix(train_xgb), label=train_xgb_lab)
xgbval<-xgb.DMatrix(data=data.matrix(train_xgb_val), label=train_xgb_val_lab)
#train the model, following a preprocessing step with a validation set
watchlist<-list(train=xgbtrain, test=xgbval)

xgb_train<-xgb.train(data=xgbtrain, max.depth=2, eta=0.1, nrounds=100, objective="binary:logistic", verbose=0, early_stopping_rounds = 20, watchlist = watchlist )
xgbpred<-predict(xgb_train, data.matrix(test_xgb))
xgbpred_c<-ifelse(xgbpred > 0.5, 1, 0)
xgbpred_c<-ifelse(xgbpred > 0.5, 1, 0)%>%factor()

cm_xgb<-confusionMatrix(xgbpred_c, as.factor(test_lab))

xgb_importance<-xgb.importance(model=xgb_train)
xgb.plot.importance(xgb_importance)


xgb_test<-xgboost(data=xgbtrain, max.depth=2, nrounds=xgb_train$best_iteration, verbose=0, objective="binary:logistic")
xgbpred_t<-predict(xgb_test, data.matrix(test_xgb))
xgbpred_t1<-ifelse(xgbpred_t>0.5, 1, 0)

cm_xgbpred_t1<-confusionMatrix(as.factor(xgbpred_t1), as.factor(test_lab))
cm_xgbpred_t1

Results<-add_row(Results, Model="xgb", Accuracy=cm_xgbpred_t1$overall["Accuracy"])
kable(Results, format="pipe", caption="Model Accuracy")




#NNet

#install.packages("nnet")
set.seed(1967, sample.kind = "Rounding")
#library(nnet)
##default parameters nnet supports only one hidden layer 
nn_model<-nnet(as.factor(HeartDisease)~., data=HDS_train, size=5, linout=FALSE)
nn_pred<-predict(nn_model, newdata=HDS_test, type="class")
nn_pred<-predict(nn_model, newdata=HDS_test, type="class")%>%factor()
nn_cm<-confusionMatrix(nn_pred, as.factor(HDS_test$HeartDisease))


accuracy_nn<-sum(nn_pred==HDS_test$HeartDisease)/ nrow(HDS_test)



Results<-add_row(Results, Model="nnet default", Accuracy=accuracy_nn)
kable(Results, format="pipe", caption="Model Accuracy")






#training grid of node size and weigh decay for regularization  and avoiding overfitting 
grid<-expand.grid(size=seq(1,5,2), decay=seq(0,0.1,0.02))
                  
#tuning parameters of 3-fold cross validation 
ctl<-trainControl(method="cv", 
                  number=3, search="grid")
#scaling numeric variables with min-max normalization 
nn_train<-train(as.factor(HeartDisease)~., data=HDS_train, preProcess="range", method="nnet", trControl=ctl, tuneGrid=grid, trace=FALSE)

max(nn_train$results$Accuracy)
plot(nn_train)
pred_train<-predict(nn_train, HDS_train)
cm_train<-confusionMatrix(pred_train, as.factor(HDS_train$HeartDisease))
cm_train
pred_test<-predict(nn_train, HDS_test)
cm_test<-confusionMatrix(pred_test, as.factor(HDS_test$HeartDisease))
cm_test
#to check for overfitting
Results<-add_row(Results, Model="nnet", Accuracy=cm_test$overall["Accuracy"])
kable(Results, format="pipe", caption="Model Accuracy")
#check over fitting
checkNNetModel<-cbind(cm_train$overall, cm_test$overall)
kable(checkNNetModel, format="pipe", col.names = c("Overall", "Train", "Test"), caption = "overfitting check")

#varImp(nn_train)
n_imp<-varImp(nn_train)
#barplot(n_imp$importance$Overall, names.arg = rownames(n_imp$importance))
kable(n_imp$importance)




### Random FOrests


#default parameters with 
RF_train<-randomForest(as.factor(HeartDisease) ~ ., data=HDS_train, ntree=500)

Predict_RF<-predict(RF_train, HDS_test)

RF_cm<-confusionMatrix(Predict_RF, as.factor(HDS_test[,12]))
RF_cm
Results<-add_row(Results, Model="RF_test", Accuracy=RF_cm$overall["Accuracy"])
kable(Results, format="pipe", caption="Model Accuracy")
#variable importance

imp_rf<-varImp(RF_train)
#graph
varImpPlot(RF_train)
kable(imp_rf)




#Tuning the parameters
grid<-data.frame(mtry=c(1,2,5,10))
RF_opt<-train(as.factor(HeartDisease)~., data=HDS_train, method="rf", ntree=500, trControl=trainControl(method="cv", number=5), tuneGrid=grid)
plot(RF_opt)

Pred_RF_O<-predict(RF_opt, HDS_test)
cm_RF_O<-confusionMatrix(Pred_RF_O, as.factor(HDS_test[,12]))

Results<-add_row(Results, Model="RF_trained", Accuracy=cm_RF_O$overall["Accuracy"])

kable(Results, fomrat="pipe", caption="Model Accuracy")
