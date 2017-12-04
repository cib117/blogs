#######################################################################################
# Clear workspace and set seed
#######################################################################################
rm(list=ls())
set.seed(1804)
#######################################################################################
# Read in required packages
#######################################################################################
library(mlr)
library(dplyr)
library(ggplot2)
library(stringr)
library(lime)
#######################################################################################
# Read in FIFA 18 data
#######################################################################################
df <- read.csv('CompleteDataset.csv', head=T, stringsAsFactors = F)
#######################################################################################
# Data Preprocessing
#######################################################################################
# Convert wage variable to numeric
df$Wage <- as.numeric(str_match(df$Wage, '[0-9]+'))

# Convert value variable to numeric
# Value in Millions
# So divide K values by 1000
df$Value <- ifelse(str_detect(df$Value, 'K'),
                    as.numeric(str_match(df$Value, '[0-9]+'))/1000,
                    as.numeric(str_match(df$Value, '[0-9]+')))

# Create primary position variable
# I treat wingers as attackers
df$primarypos <- str_match(df$Preferred.Positions, '[A-Z]+')
df$primarypos[str_detect(df$primarypos, 'ST|RW|LW|CF')] <- 'Attacker'
df$primarypos[str_detect(df$primarypos, 'CDM|RM|CM|LM|CAM')] <- 'Midfielder'
df$primarypos[str_detect(df$primarypos, 'CB|LB|RB|RWB|LWB')] <- 'Defender'
df$primarypos[str_detect(df$primarypos, 'GK')] <- 'Goalkeeper'
df$primarypos <- as.factor(df$primarypos)

# Exclude players with zero salary
df <- df[df$Wage!=0,]

# Plot players' wages
ggplot(df, aes(Wage)) +
  geom_histogram() +
  theme_bw() +
  ggtitle("Distribution of Players' Wages") +
  xlab('Wage') +
  ylab('Count')

# Create logged version of wages
df$logWage <- log(df$Wage)
#######################################################################################
# Build Model with MLR
#######################################################################################
# randomly shuffle the data
df <- df[sample(nrow(df)),]

# split into training and test splits
train <- df[1:12000,]
test <- df[12001:nrow(df),]

trainsub <- train[, c('Value', 'Overall', 'primarypos', 'Age', 'logWage')]
testsub <- test[, c('Value', 'Overall', 'primarypos', 'Age', 'logWage')]

# make task
tsk = makeRegrTask(data=trainsub, target = "logWage")

# Tune hyperparameters of model
# hidden layer size
discrete_ps = makeParamSet(
  makeDiscreteParam("ntree", values = c(500, 1000))
)
# make grid
ctrl = makeTuneControlGrid()
# use 10-fold cross validation
rdesc = makeResampleDesc("CV", iters = 10L)
# Tune the parameters
res = tuneParams("regr.randomForest", task = tsk, resampling = rdesc,
                 par.set = discrete_ps, control = ctrl)
# Retrain model
lrn = setHyperPars(makeLearner("regr.randomForest"), par.vals = res$x)
mod = train(lrn, tsk)
predictions <- predict(mod, newdata=testsub)
test$preds <- predictions$data$response
#######################################################################################
# Interpret Model with different methods using MLR
#######################################################################################
###########################
# Permutation importance
###########################
varimp <- generateFilterValuesData(tsk, imp.learner=lrn, method = "permutation.importance")
# plot importance
plotFilterValues(varimp) +
  ggtitle('Feature Importance') +
  xlab('Feature') +
  scale_x_discrete(labels=c('primarypos' = 'Position',
                            'Value' = 'Transfer Value',
                            'Overall'= 'Ability'))  +
  ylab('Predictive Importance') +
  theme_bw()
###########################
# Partial Dependence
###########################
# PD for overall
pd.regr <- generatePartialDependenceData(mod, tsk, c("Overall"), 
                                         fun = function(x) quantile(x, c(.05, .5, .95)))
plotPartialDependence(pd.regr) +
  ggtitle('Wages and Player Ability') +
  xlab('Player Ability') +
  ylab('Wages (logged)') +
  theme_bw()

# PD for value
pd.regr <- generatePartialDependenceData(mod, tsk, "Value", 
                                         fun = function(x) quantile(x, c(.05, .5, .95)))
plotPartialDependence(pd.regr) +
  ggtitle('Wages and Transfer Value') +
  xlab('Transfer Value (m)') +
  ylab('Wages (logged)') +
  theme_bw()
###########################
# LIME
###########################
test<- test[with(test, order(-Overall)), ]
testsub <- testsub[with(testsub, order(-Overall)), ]
explainer <- lime(trainsub, mod)
forexplain <- testsub[1:6,1:4]
# set rownames to name to make plot easier to read
rownames(forexplain) <- test$Name[1:6]
explanations <- explain(forexplain, explainer, n_features = 4)
plot_features(explanation = explanations)