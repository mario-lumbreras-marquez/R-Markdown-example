#################################################
# Random forest prediction model
#################################################

# Remove id column
df_ml = subset(twins_all_bwh, select = -c(study_id))
print(names(df_ml))
df<-df_ml

# Create factors
df$race<-factor(df$race)
df$art<-factor(df$art)
df$htn<-factor(df$htn)
df$diabetes<-factor(df$diabetes)
df$anomaly<-factor(df$anomaly)
df$chorion<-factor(df$chorion)
df$adm_presb<-factor(df$adm_presb)
df$stat_adm<-factor(df$stat_adm)
df$delivery_final<-factor(df$delivery_final)

# Missing data plot
library(naniar)
vis_miss(df)

# RF imputation
library(missForest)
df <- as.data.frame(df) 
set.seed(222)
twins.imputed <- missForest(df, verbose = T)
twins.imputed$OOBerror
twins.bwh <- twins.imputed$ximp
vis_miss(twins.bwh)

# Creating training and test sets
library(caret)
set.seed(12341)
Train <- createDataPartition(twins.bwh$delivery_final, p=0.8, list=FALSE)
twins.train <- twins.bwh[ Train, ]
twins.test <- twins.bwh[ -Train, ]

# Load libraries
library(rpart)
library(rpart.plot)
library(adabag)
library(randomForest)
library(MASS)
library(gbm)
library(ROCR)
library(rfUtilities)
library(ResourceSelection)

# Recursive feature elimination - variable selection
# Used to to repeatedly construct a model (10-CV) and remove features with low weights
set.seed(1281)
library(mlbench)
library(caret)

# Define the control using a random forest selection function
control <- rfeControl(functions=rfFuncs, method="cv", number=10, repeats = 5)

# Run the RFE algorithm
results <- rfe(twins.train[,(-17)], as.factor(t(twins.train[,17])), sizes=c(1:18), rfeControl=control)

# Summarize the results
print(results)

# List the chosen features
predictors(results)

# Plot the results
plot(results, type=c("g", "o"))

# Fit
results$fit

# Save the RF 
twins.train.rf.all<-results$fit

# Size of trees (number of nodes) in an ensemble
hist(treesize(twins.train.rf.all))

# Plotting the model shows us that after about 300 trees, not much changes in terms of error (OOB) 
# It fluctuates a bit but not to a large degree
plot(twins.train.rf.all$err.rate[,1])

# Printing the model shows the number of variables tried at each split to be 3 and 
# an OOB estimate of error rate 23.23%. 
print(twins.train.rf.all)

# Predictor importance
varImpPlot(twins.train.rf.all, sort = T, n.var = 10, 
           main = "Variable Importance", 
           col=300,
           cex.main=1.5, cex.lab=1.25, cex.axis=0.75)

# Variable Importance - MeanDecreaseGini
var.imp = data.frame(importance(twins.train.rf.all,  
                                type=2))
# Make row names as columns
var.imp$Variables = row.names(var.imp)  
print(var.imp[order(var.imp$MeanDecreaseGini,decreasing = T),])

# Variable Importance - MeanDecreaseAccuracy
var.imp = data.frame(importance(twins.train.rf.all,  
                                type=1))

# Make row names as columns
var.imp$Variables = row.names(var.imp)  
print(var.imp[order(var.imp$MeanDecreaseAccuracy,decreasing = T),])

# Calculating predicted probability for the TEST dataset
library(pROC)
twins.test$pred.rf.all<-predict(twins.train.rf.all,newdata=twins.test,type="prob")[,2]

# Calculating the area under the ROC curve TEST dataset
performance(prediction(twins.test$pred.rf.all, twins.test$delivery_final), "auc")
plot.roc(twins.test$delivery_final, twins.test$pred.rf.all, print.auc=T, ci=T,
         main="Figure 1. C-statistic",
         col="28", lwd="5", legacy.axes=T,
         asp=NA, cex.main=1.5, cex.lab=1.25, cex.axis=0.75)

# Confusion matrix TEST - 0.5 threshold
twins.test$predicted.response = predict(twins.train.rf.all, twins.test)
print(confusionMatrix(data = twins.test$predicted.response, reference = twins.test$delivery_final, positive = "1"))

# Calibration plot
library(gbm)
summary(twins.test$pred.rf.all)
calibrate.plot(twins.test$delivery_final, twins.test$pred.rf.all,
               main="Figure 2. Calibration plot-Relationship between predicted and observed",
               xlim=c(0.1600, 0.9960), ylim=c(0,1.0),
               cex.main=1.5, cex.lab=1.25, cex.axis=0.75)

# Conditional permutation importance
set.seed(2020)
readingSkills.rf <- randomForest::randomForest(delivery_final ~ effac_adm + vd_prior +
                                                 usb_efw + usa_efw + gestage + dil_adm +
                                                 mat_age + bmi_last + tbw_discordance +
                                                 stat_adm, data = twins.train, 
                                               mtry = 3, ntree = 500, importance = TRUE, 
                                               keep.forest = TRUE, keep.inbag = TRUE)


set.seed(29087)
library(permimp)
var_imp_rf <- permimp(readingSkills.rf, conditional = TRUE, threshold = .95, do_check = FALSE)
var_imp_rf

library("RColorBrewer")
plot(var_imp_rf, type = "bar",
     main = "Figure 1. Variable importance-Conditional permutation importance",
     sub = "Threshold = 0.95",
     cex.main=1.5, cex.lab=1.25, cex.axis=1.0,
     legend = TRUE,
     args.legend = list(x = "bottomright", cex = .85, title="Predictor"),
     horiz = TRUE,
     col = brewer.pal(n = 10, name = "RdBu"))

#################################################
# End
#################################################
