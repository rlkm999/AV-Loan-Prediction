#clear environment
rm(list = ls())

#load libraries
library(caret)
library(DMwR)
library(MASS)
library(randomForest)
library(e1071)
library(caretEnsemble)
library(rpart)
library(xgboost)

#set path
setwd("G:/AV/loan prediction")

#read files
train<-read.csv("train_u6lujuX_CVtuZ9i.csv")
test<-read.csv("test_Y3wMUE5_7gLdaTN.csv")


#data preprocessing
str(train)
str(test)
summary(train)
summary(test)

train$Credit_History<-as.factor(train$Credit_History)
test$Credit_History<-as.factor(test$Credit_History)

Loan_ID<-test$Loan_ID
Loan_ID<-as.data.frame(Loan_ID)
train$Loan_ID<-NULL
test$Loan_ID<-NULL

colSums(is.na(train))
colSums(is.na(test))

train<-knnImputation(train,k=5)
test<-knnImputation(test,k=5)
######### Boxplots
boxplot(train$ApplicantIncome)
mean.ApplicantIncome<-mean(train$ApplicantIncome)
upperwhisker<-boxplot.stats(train$ApplicantIncome)$stats[5]
train$ApplicantIncome[train$ApplicantIncome>upperwhisker]<-mean.ApplicantIncome

boxplot(test$ApplicantIncome)
mean.ApplicantIncome.test<-mean(test$ApplicantIncome)
upperwhisker.test<-boxplot.stats(test$ApplicantIncome)$stats[5]
test$ApplicantIncome[test$ApplicantIncome>upperwhisker.test]<-mean.ApplicantIncome.test
# f<-function(x){
#   q<-quantile(x,c(.05,.095))
#   x[x>q[2]]<-q[2]
#   
# }
# 
# f(train$ApplicantIncome)->train$ApplicantIncome
# boxplot(train$ApplicantIncome)
boxplot(train$CoapplicantIncome)
mean.CoapplicantIncome<-mean(train$CoapplicantIncome)
upperwhisker1<-boxplot.stats(train$CoapplicantIncome)$stats[5]
train$CoapplicantIncome[train$CoapplicantIncome>upperwhisker1]<-mean.CoapplicantIncome

boxplot(test$CoapplicantIncome)
mean.CoapplicantIncome.test<-mean(test$CoapplicantIncome)
upperwhisker1.test<-boxplot.stats(test$CoapplicantIncome)$stats[5]
test$CoapplicantIncome[test$CoapplicantIncome>upperwhisker1.test]<-mean.CoapplicantIncome.test

#########
boxplot(train$LoanAmount)
mean.LoanAmount<-mean(train$LoanAmount)
upperwhisker2<-boxplot.stats(train$LoanAmount)$stats[5]
train$LoanAmount[train$LoanAmount>upperwhisker2]<-mean.LoanAmount

boxplot(test$LoanAmount)
mean.LoanAmount.test<-mean(test$LoanAmount)
upperwhisker2.test<-boxplot.stats(test$LoanAmount)$stats[5]
test$LoanAmount[test$LoanAmount>upperwhisker2.test]<-mean.LoanAmount.test
#########

set.seed(100)
transform<-preProcess(train,method = c('center','scale'))
std.train<-predict(transform,train)
std.test<-predict(transform,test)

summary(std.test)
table(std.test$Dependents)
std.test$Dependents[std.test$Dependents==""]<-"0"
std.test$Dependents<-factor(std.test$Dependents,levels = c("0","1","2","3+"))

std.train$Dependents[std.train$Dependents==""]<-"0"
std.train$Dependents<-factor(std.train$Dependents,levels = c("0","1","2","3+"))

table(std.test$Self_Employed)
std.test$Self_Employed[std.test$Self_Employed==""]<-"No"
std.test$Self_Employed<-factor(std.test$Self_Employed,levels = c("No","Yes"))

std.train$Self_Employed[std.train$Self_Employed==""]<-"No"
std.train$Self_Employed<-factor(std.train$Self_Employed,levels = c("No","Yes"))

std.test$Gender[std.test$Gender==""]<-"Male"
std.test$Gender<-factor(std.test$Gender,levels = c("Female","Male"))

std.train$Gender[std.train$Gender==""]<-"Male"
std.train$Gender<-factor(std.train$Gender,levels = c("Female","Male"))

std.train$Married[std.train$Married==""]<-"Yes"
std.train$Married<-factor(std.train$Married,levels = c("No","Yes"))

std.train<-knnImputation(std.train,k=5)
# levels(std.train$Loan_Status)<-c(levels(std.train$Loan_Status),"0","1")
# 
# std.train$Loan_Status[std.train$Loan_Status=="Y"]<-"1"
# std.train$Loan_Status[std.train$Loan_Status=="N"]<-"0"
# std.train$Loan_Status<-factor(std.train$Loan_Status,levels = c(0,1))
str(std.train)
str(std.test)

summary(std.train)
summary(std.test)
names(std.train)

predictors<-c("LoanAmount","Loan_Amount_Term","Credit_History","ApplicantIncome","CoapplicantIncome")
outcome<-c("Loan_Status")

#Model building

#logistic
glm.model<-glm(std.train$Loan_Status~.,data = std.train,family = "binomial")
summary(glm.model)

step.model<-stepAIC(glm.model)

p<-predict(step.model,std.train,type = "response")

ifelse(p>0.5,"Y","N")
tab<-table(p>0.5,std.train$Loan_Status)
tab

acc<-sum(diag(tab))/sum(tab)
acc

Loan_Status<-predict(step.model,std.test,type = "response")
Loan_Status<-ifelse(Loan_Status>0.5,"Y","N")
table(Loan_Status)

#Random forest
rf.model<-randomForest(std.train$Loan_Status~.,data = std.train,ntree=200,cp=0.0055)
summary(rf.model)
plot(rf.model)

pred<-predict(rf.model,std.train)
confusionMatrix(pred,std.train$Loan_Status)

traincv<-trainControl(method = "repeatedcv",number = 3,repeats = 3)
model_rf<-train(std.train[,predictors],std.train[,outcome],method="rf",trControl=traincv,tuneLength=10)

pf<-predict(model_rf,std.train)
confusionMatrix(pf,std.train$Loan_Status)

Loan_Status<-predict(model_rf,std.test)
Loan_Status
table(Loan_Status)

#c5.0
c5.0.model<-C5.0(x=std.train[,-12],y=std.train[,12])
summary(c5.0.model)
plot(c5.0.model)

pd<-predict(c5.0.model,std.train[,-12])
table(pd)

confusionMatrix(pd,std.train$Loan_Status)

Loan_Status<-predict(c5.0.model,std.test)
table(Loan_Status)

#SVM
svm.model<-svm(std.train$Loan_Status~.,data = std.train)

pred.svm<-predict(svm.model,std.train)
confusionMatrix(pred.svm,std.train$Loan_Status)

Loan_Status<-predict(svm.model,std.test)
Loan_Status
table(Loan_Status)

#GBM
traincv<-trainControl(method = "repeatedcv",number = 3,repeats = 3)

gbm<-train(std.train[,-12],std.train$Loan_Status,method="gbm",trControl=traincv,tuneLength=10)
pred.gbm<-predict(gbm,std.train)
confusionMatrix(pred.gbm,std.train$Loan_Status)

Loan_Status<-predict(gbm,std.test)
Loan_Status
table(Loan_Status)

#Xgb
xgb.ctrl <- trainControl(method = "repeatedcv", repeats = 3, number = 3, search='random', allowParallel=T)

xgb.tune <-train(Loan_Status~.,data = std.train,method="xgbTree",trControl=xgb.ctrl,tuneLength=20,verbose=T,metric="Accuracy",nthread=3)

px<-predict(xgb.tune,std.train)
confusionMatrix(px,std.train$Loan_Status)

Loan_Status<-predict(xgb.tune,std.test)
table(Loan_Status)


# #Ensemble
# methodList=c('glm','rf')
# 
# traincv<-trainControl(method = "repeatedcv",number = 3,repeats = 3,classProbs = T)
# model_list <- caretList(Loan_Status~., data=std.train,
#                         trControl=traincv,
#                         methodList=methodList)
# 
# ensemble<-caretEnsemble(model_list)
# 
# pe<-predict(ensemble,std.train)
# confusionMatrix(pe,std.train$Loan_Status)
# 
# Loan_Status<-predict(ensemble,std.test)
# Loan_Status
# table(Loan_Status)


Loan_Status<-as.data.frame(Loan_Status)

Submission<-cbind(Loan_ID,Loan_Status)
write.csv(Submission,file = "Submission.csv",row.names = F)
