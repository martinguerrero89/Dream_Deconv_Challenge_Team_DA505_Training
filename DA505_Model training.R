#Starting from data of Merge_data.R

#Input data must be a dataset with tpm data with samples in columns and genes in rows

data=readRDS("./input/train_samples.rds")

#load(file="./Data/featureMaster.Rdata")

#Metadata must have a column with at least "samplename", "batch", and "coarsegrain" and "finegrain" labels

metadata = read.table("./input/Deconv_training_datasets_tidy.txt",
                      sep = "\t",
                      header = TRUE,
                      stringsAsFactors = FALSE
)

data = log2(data + 1) #Not strictly needed

source("./functions/Rhino_norm.R")

X= apply(data,2,mynorm4)

rm(data)


# CG admixtures -----------------------------------------------------------


mdtrain = metadata[metadata$coarsegrain != "other" & metadata$coarsegrain != "noise",]


Trainsamples= NULL
Testsamples= NULL
set.seed(314)
for(i in unique(mdtrain$coarsegrain)){
  len= sum(mdtrain$coarsegrain==i)*0.75
  w= which(mdtrain[,"coarsegrain"]==i)
  trains=sample(w,len,replace=FALSE)
  tests= w[!w %in% trains]
  Trainsamples=c(Trainsamples,trains)
  Testsamples=c(Testsamples,tests)
}

Trainsamples=  mdtrain$samplename[Trainsamples]
Testsamples= mdtrain$samplename[Testsamples]

# Mixinig ALL

# Create Coarsegrain sets

#Trainset
TRAINSET = "Train"
Xtrain = X[,Trainsamples]

mdtrain= metadata[Trainsamples,]
Types = unique(mdtrain$coarsegrain)
Type = mdtrain$coarsegrain

nusamples = 2000
props = matrix(runif(length(Types) * nusamples), ncol = nusamples)

M= matrix(rep(NA,length(Types) * nusamples),ncol=nusamples)
binom_prob= seq(0.1,0.7,by=0.1)
for(i in 1:ncol(M)){
  bp= sample(binom_prob,1)
  binom= rbinom(nrow(M), 1, bp)
  M[,i]=binom
}

while(any(colSums(M) == 0)){
  ind=which(colSums(M)==0)
  for(j in ind){
    bp= sample(binom_prob,1)
    binom= rbinom(nrow(M), 1, bp)
    M[,j]=binom
    }
}

props = props * M
rm(M)

props = apply(props, 2, function(x) x / sum(x))

exp = matrix(rep(0, nrow(Xtrain) * ncol(props)), ncol = ncol(props))

for (j in 1:ncol(props)) {
  for (i in 1:length(Types)) {
    if(props[i, j]==0){next}
    #select cell type to mix
    s = which(Type == Types[i])
    
    #Decide how many samples of the cell type are going to be mixed
    numofsamples= sample(1:length(s),1)
    
    #Determine sampling probabilities according batch and finegrain type
    Factor= paste(mdtrain[s,"batch"],mdtrain[s,"finegrain"],sep=".")
    sampling_prob= (1/length(unique(Factor))) /table(Factor)
    sampling_prob= sampling_prob[Factor]
    l= sample(s,numofsamples,replace=FALSE,prob=sampling_prob)
    
    sampleprop = runif(numofsamples)
    sampleprop = sampleprop / sum(sampleprop)
    EXPi = rowSums(t(t(Xtrain[, l]) * sampleprop))
    exp[, j] = exp[, j] + EXPi * props[i, j]
  }
}


train = exp
trainprop = props
colnames(train) = 1:ncol(train)
rownames(train) = rownames(Xtrain)
rownames(trainprop) = Types

#Test set
TESTSET = "Test"
Xtest = X[, Testsamples]

mdtest = metadata[Testsamples,]

Types = unique(mdtest$coarsegrain)
Type = mdtest$coarsegrain


nusamples = 1000
props = matrix(runif(length(Types) * nusamples), ncol = nusamples)

M= matrix(rep(NA,length(Types) * nusamples),ncol=nusamples)
binom_prob= seq(0.1,0.7,by=0.1)
for(i in 1:ncol(M)){
  bp= sample(binom_prob,1)
  binom= rbinom(nrow(M), 1, bp)
  M[,i]=binom
}

while(any(colSums(M) == 0)){
  ind=which(colSums(M)==0)
  for(j in ind){
    bp= sample(binom_prob,1)
    binom= rbinom(nrow(M), 1, bp)
    M[,j]=binom
  }
}

props = props * M
rm(M)
props = apply(props, 2, function(x)
  x / sum(x))

exp = matrix(rep(0, nrow(Xtrain) * ncol(props)), ncol = ncol(props))


for (j in 1:ncol(props)) {
  for (i in 1:length(Types)) {
    if(props[i, j]==0){next}
    #select cell type to mix
    s = which(Type == Types[i])
    
    #Decide how many samples of the cell type are going to be mixed
    numofsamples= sample(1:length(s),1)
    
    #Determine sampling probabilities according batch and finegrain type
    Factor= paste(mdtest[s,"batch"],mdtest[s,"finegrain"],sep=".")
    sampling_prob= (1/length(unique(Factor))) /table(Factor)
    sampling_prob= sampling_prob[Factor]
    l= sample(s,numofsamples,replace=FALSE,prob=sampling_prob)
    
    sampleprop = runif(numofsamples)
    sampleprop = sampleprop / sum(sampleprop)
    EXPi = rowSums(t(t(Xtest[, l]) * sampleprop))
    exp[, j] = exp[, j] + EXPi * props[i, j]
  }
}



test = exp
testprop = props

colnames(test) = 1:ncol(test)
rownames(test) = rownames(Xtrain)

rownames(testprop) = Types


#Train Coarse GLMnet model ----

library(glmnet)
library(foreach)
library(tibble)
library(doParallel)
library(dplyr)
library(optparse)
registerDoParallel(cores=6)




alpha_range <- c(0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9)
lambda_ratio_range <- c(10e-8,10e-7,10e-6,10e-5,10e-4,10e-3)


option_list <- list(
  make_option("--experimenttag", action="store", type="character", default="default-experiment", help = "Set experiment tag id "),
  make_option("--nfeature", action="store", type="numeric", default=500, help = "Set the maximun number of features ")
)
opt <- parse_args(OptionParser(option_list=option_list))

parms <- expand.grid(alpha = alpha_range, lambda_ratio = lambda_ratio_range)
results_final <-c()
results_final_models <- list()

# Starting training  ---------
for (label_number in rownames(trainprop)) {
  
  labels <- trainprop[label_number, ]
  trainset<- t(train)
  labels_test <- testprop[label_number, ]
  testset <- t(test)
  
  # Start parallel fine tuning  -----------
  results <- foreach(i = 1:nrow(parms),.packages = "glmnet",.combine = rbind) %dopar% {
    alpha <- parms[i,]$alpha
    lambda_ratio <- parms[i,]$lambda_ratio
    model <- cv.glmnet(
      type.measure = "mse",
      nfolds = 5,
      y = labels,
      x = trainset,
      alpha = alpha,
      standardize = FALSE,
      nlambda = 100,
      lambda.min.ratio = lambda_ratio,
      family = "gaussian",
      lower.limit = -Inf
    )
    
    preds <- predict(model, testset,s = "lambda.1se")
    spear <- cor(x = preds, y = labels_test, method = "spearman")
    pears <- cor(x = preds, y = labels_test, method = "pearson")
    partial_results<-data.frame(label_number,parms[i,], pearson = pears, spearman = spear)
    partial_results 
    
  }
  # Select best model ---------
  best_model <- results %>% arrange(desc(pearson)) %>% filter(row_number()==1) 
  print(paste("selecting best model for ",best_model$label_number," : ",best_model$alpha, ", ", best_model$lambda_ratio," Pearson value : ", best_model$pearson,sep=""))
  trainset <- rbind(trainset,testset)
  labels <- c(labels,labels_test)
  model <- cv.glmnet(
    type.measure = "mse",
    nfolds = 5,
    y = labels,
    x = trainset,
    alpha = best_model$alpha,
    standardize = FALSE,
    nlambda = 100,
    lambda.min.ratio = best_model$lambda_ratio,
    family = "gaussian",
    lower.limit = -Inf
  )
  results_final_models[[label_number]]<-model
  save(results_final_models,file = paste0("glmnet_model_coarsegrained",opt$experimenttag,".rdata"),compress = "gzip")
  results_final<-rbind(results_final,results)
  readr::write_csv(results_final,path=paste0("glmnet_model_coarsegrained_model_selection",opt$experimenttag,".csv"))
}

#FG admixtures ----

mdtrain = metadata[metadata$finegrain != "other" & metadata$finegrain != "noise",]
#mdtrain = metadata[metadata$finegrain != "other",] 

Trainsamples= NULL
Testsamples= NULL
set.seed(314)
for(i in unique(mdtrain$finegrain)){
  len= sum(mdtrain$finegrain==i)*0.75
  w= which(mdtrain[,"finegrain"]==i)
  trains=sample(w,len,replace=FALSE)
  tests= w[!w %in% trains]
  Trainsamples=c(Trainsamples,trains)
  Testsamples=c(Testsamples,tests)
}

Trainsamples=  mdtrain$samplename[Trainsamples]
Testsamples= mdtrain$samplename[Testsamples]
#Mixinig ALL


#finegrain

#Trainset
TRAINSET = "Train"
Xtrain = X[,Trainsamples]

mdtrain= metadata[Trainsamples,]
Types = unique(mdtrain$finegrain)
Type = mdtrain$finegrain

nusamples = 2000
props = matrix(runif(length(Types) * nusamples), ncol = nusamples)

M= matrix(rep(NA,length(Types) * nusamples),ncol=nusamples)
binom_prob= seq(0.1,0.7,by=0.1)
for(i in 1:ncol(M)){
  bp= sample(binom_prob,1)
  binom= rbinom(nrow(M), 1, bp)
  M[,i]=binom
}

if(any(colSums(M) == 0)){
  ind=which(colSums(M)==0)
  for(j in ind){
    ind2=sample(1:nrow(M),1)
    M[ind2,j]=1
  }
}

props = props * M
rm(M)

props = apply(props, 2, function(x) x / sum(x))

exp = matrix(rep(0, nrow(Xtrain) * ncol(props)), ncol = ncol(props))

for (j in 1:ncol(props)) {
  for (i in 1:length(Types)) {
    if(props[i, j]==0){next}
    #select cell type to mix
    s = which(Type == Types[i])
    
    #Decide how many samples of the cell type are going to be mixed
    numofsamples= sample(1:length(s),1)
    
    #Determine sampling probabilities according batch and finegrain type
    Factor= paste(mdtrain[s,"batch"],mdtrain[s,"finegrain"],sep=".")
    sampling_prob= (1/length(unique(Factor))) /table(Factor)
    sampling_prob= sampling_prob[Factor]
    l= sample(s,numofsamples,replace=FALSE,prob=sampling_prob)
    
    sampleprop = runif(numofsamples)
    #sampleprop = runif(sum(Type == Types[i]))
    sampleprop = sampleprop / sum(sampleprop)
    EXPi = rowSums(t(t(Xtrain[, l]) * sampleprop))
    exp[, j] = exp[, j] + EXPi * props[i, j]
  }
}


train = exp
trainprop = props
colnames(train) = 1:ncol(train)
rownames(train) = rownames(Xtrain)
rownames(trainprop) = Types

#Test set
TESTSET = "Test"
Xtest = X[, Testsamples]

mdtest = metadata[Testsamples,]

Types = unique(mdtest$finegrain)
Type = mdtest$finegrain


nusamples = 1000
props = matrix(runif(length(Types) * nusamples), ncol = nusamples)

M= matrix(rep(NA,length(Types) * nusamples),ncol=nusamples)
binom_prob= seq(0.1,0.7,by=0.1)
for(i in 1:ncol(M)){
  bp= sample(binom_prob,1)
  binom= rbinom(nrow(M), 1, bp)
  M[,i]=binom
}

if(any(colSums(M) == 0)){
  ind=which(colSums(M)==0)
  for(j in ind){
    ind2=sample(1:nrow(M),1)
    M[ind2,j]=1
  }
}

props = props * M
rm(M)
props = apply(props, 2, function(x)
  x / sum(x))

exp = matrix(rep(0, nrow(Xtrain) * ncol(props)), ncol = ncol(props))

#for (j in 1:ncol(props)) {
#  for (i in 1:length(Types)) {
#    s = which(Type == Types[i])
#    sampleprop = runif(sum(Type == Types[i]))
#    sampleprop = sampleprop / sum(sampleprop)
#    EXPi = rowSums(t(t(Xtest[, s]) * sampleprop))
#    exp[, j] = exp[, j] + EXPi * props[i, j]
#  }
#}


for (j in 1:ncol(props)) {
  for (i in 1:length(Types)) {
    if(props[i, j]==0){next}
    #select cell type to mix
    s = which(Type == Types[i])
    
    #Decide how many samples of the cell type are going to be mixed
    numofsamples= sample(1:length(s),1)
    
    #Determine sampling probabilities according batch and finegrain type
    Factor= paste(mdtest[s,"batch"],mdtest[s,"finegrain"],sep=".")
    sampling_prob= (1/length(unique(Factor))) /table(Factor)
    sampling_prob= sampling_prob[Factor]
    l= sample(s,numofsamples,replace=FALSE,prob=sampling_prob)
    
    sampleprop = runif(numofsamples)
    #sampleprop = runif(sum(Type == Types[i]))
    sampleprop = sampleprop / sum(sampleprop)
    EXPi = rowSums(t(t(Xtest[, l]) * sampleprop))
    exp[, j] = exp[, j] + EXPi * props[i, j]
  }
}



test = exp
testprop = props

colnames(test) = 1:ncol(test)
rownames(test) = rownames(Xtrain)

rownames(testprop) = Types

#save(list= c("train","trainprop","test","testprop"),file="deconv_fgdata_cps_new_feat_last3.RData")  
#load("deconv_fgdata_cps_new_feat_last3.RData")  


# Train Fine GLMnet model----


#library(caret)
library(glmnet)
library(foreach)
library(tibble)
#library(randomForest)
library(doParallel)
library(dplyr)
library(optparse)
registerDoParallel(cores=4)




alpha_range <- c(0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9)
lambda_ratio_range <- c(10e-8,10e-7,10e-6,10e-5,10e-4,10e-3)
#lambda_ratio_range <- c(10e-5,10e-4,10e-3)

#### MAIN 

option_list <- list(
  make_option("--experimenttag", action="store", type="character", default="default-experiment", help = "Set experiment tag id "),
  make_option("--nfeature", action="store", type="numeric", default=500, help = "Set the maximun number of features ")
)
opt <- parse_args(OptionParser(option_list=option_list))



parms <- expand.grid(alpha = alpha_range, lambda_ratio = lambda_ratio_range)
results_final <-c()
results_final_models <- list()

# Starting training  ---------
for (label_number in rownames(trainprop)) {

  labels <- trainprop[label_number, ]
  trainset<- t(train)
  labels_test <- testprop[label_number, ]
  testset<- t(test)
  
  # Start parallel fine tuning  -----------
  results <- foreach(i = 1:nrow(parms),.packages = "glmnet",.combine = rbind) %dopar% {
    alpha <- parms[i,]$alpha
    lambda_ratio <- parms[i,]$lambda_ratio
    model <- cv.glmnet(
      type.measure = "mse",
      nfolds = 5,
      y = labels,
      x = trainset,
      alpha = alpha,
      standardize = FALSE,
      nlambda = 100,
      lambda.min.ratio = lambda_ratio,
      family = "gaussian",
      lower.limit = -Inf
    )
    
    preds <- predict(model, testset,s = "lambda.1se")
    spear <- cor(x = preds, y = labels_test, method = "spearman")
    pears <- cor(x = preds, y = labels_test, method = "pearson")
    partial_results<-data.frame(label_number,parms[i,], pearson = pears, spearman = spear)
    partial_results 
    
  }
  # Select best model ---------
  best_model <- results %>% arrange(desc(pearson)) %>% filter(row_number()==1) 
  print(paste("selecting best model for ",best_model$label_number," : ",best_model$alpha, ", ", best_model$lambda_ratio," Pearson value : ", best_model$pearson,sep=""))
  trainset <- rbind(trainset,testset)
  labels <- c(labels,labels_test)
  model <- cv.glmnet(
    type.measure = "mse",
    nfolds = 5,
    y = labels,
    x = trainset,
    alpha = best_model$alpha,
    standardize = FALSE,
    nlambda = 100,
    lambda.min.ratio = best_model$lambda_ratio,
    family = "gaussian",
    lower.limit = -Inf
  )
  results_final_models[[label_number]]<-model
  save(results_final_models,file = paste0("glmnet_model_finegrained",opt$experimenttag,".rdata"),compress = "gzip")
  results_final<-rbind(results_final,results)
  readr::write_csv(results_final,path=paste0("glmnet_model_finegrained_model_selection",opt$experimenttag,".csv"))
}


