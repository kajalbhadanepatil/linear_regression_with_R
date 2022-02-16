library(caret)
library(corrplot)
library(ggplot2)
install.packages("moments",dependencies=T)
library(moments)
install.packages("lmtest",dependencies = T)
library(lmtest)
install.packages("car",dependencies=T)
library(car)
install.packages("Metrics",dependencies= T)
library(Metrics)

path = "D:/new Data science class/r_work/linear regression assesment/Dataset/PropertyTrainData.csv"

price_train = read.csv(path)

dim(price_train)
View(price_train)

path = "D:/new Data science class/r_work/linear regression assesment/Dataset/PropertyTestData.csv"

price_test = read.csv(path)

dim(price_test)
View(price_test)

# take only the features (exclude the y-variable)
cols_train = names(price_train)[- which(names(price_train) == "Price")]
cols_train

cols_test = names(price_test)[- which(names(price_test) == "Price")]
cols_test

# check for anomalies: atype: "z" -> zero, "na" -> null
checkanomalies = function(df,cols,atype)
{
  if (atype %in% c("z","na"))
  {
    features = c()
    ctr = c()
    
    for(c in cols)
    {
      if (atype == 'z')
        ctr = append(ctr,length(df[c][df[c]==0])) else
          ctr = append(ctr,length(df[c][is.na(df[c])]) )
        
        features = append(features,c) 
    }
    
    # create dataframe to store the feature and count values
    d = data.frame(feature=features,count=ctr)
  }
  else
  {
    d = "Invalid type specified. Valid values are 'z', 'na' "
  }
  
  return(d)
}

checkanomalies(price_train,cols_train,"na") # checking nulls in train 
checkanomalies(price_test,cols_test,"na") # checking nulls in test 

checkanomalies(price_train,cols_train,"z") # checking zeros in train
checkanomalies(price_test,cols_test,"z") # checking zeros in test

# The zeros in dataset seems valid. Hence we will do nothing about it.

# Plotting different charts
# distribution (histogram), outliers (boxplot), multicollinearity (heatmap)
# h:histogram, b:boxplot, m:heatmap

plotchart = function(df,cols,ctype)
{
  if (ctype %in% c('h','b','m'))
  {
    if (ctype == "m") # multicollinearity heatmap
    {
      corrmx = cor(df[cols]) # correlation matrix
      corrplot(corrmx,type='lower',method='number')
    }
    else # hist / boxplot
    {
      # histogram and boxplot
      for(c in cols)
      {
        if(ctype == "b")
          boxplot(unlist(df[c]),horizontal=T,col='red',main=c) else
            hist(unlist(df[c]),col='yellow',main=c) 
      }
    }
    msg = "success"
  }
  else
  {
    msg = paste(ctype,': Invalid Chart Type')
  }
  
  return(msg)
}

plotchart(price_train,cols_train,'h') # histogram
plotchart(price_train,cols_train,'b') # boxplot
plotchart(price_train,cols_train,'m') # heatmap

# check the normality of data
# agostino-pearson test for normality
checknormality = function(df,cols)
{
  features = c()
  status = c()
  
  for(c in cols)
  {
    res = agostino.test(unlist(df[c]))
    status = append(status,ifelse(res$p.value < 0.05, "Skewed", "Normal"))
    features = append(features,c)
  }
  
  df = data.frame(feature=features,distr=status)
  return(df)
}

checknormality(price_train,names(price_train))

# regression model
buildmodel = function(traindata,y,verbose=T)
{
  # initialise the formula (y~x)
  form = as.formula(paste(y,"~ ."))
  
  # build the linear regression model
  model = lm(form,traindata)
  
  # print the model summary, if verbose is True
  if (verbose)
    print(summary(model))
  
  # return values
  lov = list(form=form,model=model)
  return(lov)
}

y = "Price"

# model 1
lov=buildmodel(price_train,y)

# model 1 output 
f1 = lov$form
m1 = lov$model
summary(m1)

# assumptions validation 

# 1) mean of errors is 0
mean(m1$residuals)

# 2) homoscadasticity
plot(m1)

# breush-pagan test
# non constant variance test

checkhetero = function(model,ttype)
{
  if(ttype %in% c('bp','ncv'))
  {
    # BP test
    if(ttype == "bp")
    {
      ret = bptest(model)
      pvalue = ret$p.value
    } else
    {
      # NCV test
      ret = ncvTest(model)
      pvalue = ret$p
    }
    
    ret = ifelse(pvalue<0.05,"heteroscedastic model",
                 "homoscedastic model")
  } else
  {
    ret = "invalid test type"
  }
  
  return(paste(ttype,":", ret))
}

checkhetero(m1,"bp")
checkhetero(m1,"ncv")

#3) rows > cols
dim(price_train)


# Actual predictions (on the test data)
p1 = predict(m1,price_test)

# RMSE of test data
rmse(price_test$Price,p1)

# store actual and predicted data in a dataframe for analysis
df1 = data.frame('actual' = price_test$Price, 
                 'predicted' = round(p1,2))
print(df1)

# best fit line
ggplot(df1, aes(x=actual, y=predicted)) +
  geom_point(col='blue') +
  geom_smooth(method='lm',col='red') +
  ggtitle('Model 1: Actual vs Predicted')


# function for transforimg data 
transformdata = function(df,y,ttype,base=0)
{
  # get all the features of the dataset
  cols = names(df)[- which(names(df) == y)]
  
  # transform the data according to the ttype
  if (ttype == "z")
    df[cols] = apply(df[cols],2,scale) else
      if (ttype == "minmax")
        df[cols] = apply(df[cols],2,minmax) else
          if (ttype == "sqrt")
            df[cols] = sqrt(df[cols]) else
              if (ttype == "inv")
                df[cols] = 1/df[cols] else
                  if (ttype == "log")
                  {
                    if (base==0) # natural log
                      df[cols] = log(df[cols]) else
                        df[cols] = log(df[cols],base)
                  } else
                  {
                    print("::ERROR::")
                    df = paste(ttype,": Invalid type")
                  }
              return(df)
}

# z transformed data 
price_train_z = transformdata(price_train,y,"z")
print(price_train_z)

price_test_z = transformdata(price_test,y,"z")
print(price_test_z)

# model 2
lov_2 = buildmodel(price_train_z,y)

# model 2 output
f2 = lov_2$form
m2 = lov_2$model

# Actual predictions (on the test data)
p2 = predict(m2,price_test_z)

# RMSE of test data
rmse(price_test_z$Price,p2)

# store actual and predicted data in a dataframe for analysis
df2 = data.frame('actual' = price_test_z$Price, 
                 'predicted' = round(p2,2))
print(df2)


# minmax transformation
price_train_mm = transformdata(price_train,y,"minmax")
print(price_train_mm)

price_test_mm = transformdata(price_test,y,"minmax")
print(price_test_mm)

# model 3
lov_3 = buildmodel(price_train_mm,y)

# model 2 output
f3 = lov_3$form
m3 = lov_3$model

# Actual predictions (on the test data)
p3 = predict(m3,price_test_mm)

# RMSE of test data
rmse(price_test_mm$Price,p3)

# store actual and predicted data in a dataframe for analysis
df3 = data.frame('actual' = price_test_mm$Price, 
                 'predicted' = round(p3,2))
print(df3)






















