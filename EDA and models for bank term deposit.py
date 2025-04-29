import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
#warnings.filterwarnings(“ignore”)
pd.options.mode.chained_assignment = None  # default='warn'

bank_dataset = pd.read_csv('bank-direct-marketing.csv',
                   sep = ';',
                   engine = 'python')

# first step, we need to split the data into training and testing. 80% training and 20% testing.
train = bank_dataset.iloc[:32950,:]
test = bank_dataset.iloc[32950:,:]
train.head()

## taking a copy of the datasets
train_original=train.copy()
test_original=test.copy()

# Understanding the data
train.columns

# We have 21 independent variables and 1 target variable, i.e. y in the training dataset.

train.dtypes

# We can see there are three formats of data types: object, float64 and int64

train['y'].value_counts(normalize=True)
train['y'].value_counts().plot.bar()
plt.tight_layout()

# 93.6 % of training is 'No' label

# Independent Variable (Categorical)
plt.rcParams.update({'font.size': 20})

train['marital'].value_counts(normalize=True).plot.bar(title='marital')
plt.xlabel('xlabel', fontsize=4)
plt.tight_layout()

plt.show()




# Independent Variable (Ordinal)

train['education'].value_counts(normalize=True).plot.bar(figsize=(20,10), title='education')
plt.show()
train['poutcome'].value_counts(normalize=True).plot.bar(title='poutcome')
plt.show()
plt.tight_layout()

train['job'].value_counts(normalize=True).plot.bar(title='job')
plt.show()
plt.tight_layout()

#Independent Variable (Numerical)

train['emp.var.rate'].value_counts(normalize=True).plot.bar(title='emp.var.rate')
plt.show()
fig = plt.figure ()
train['nr.employed'].value_counts(normalize=True).plot.bar(title='nr.employed')
plt.tight_layout()
plt.show()



sns.distplot(train['nr.employed'])
plt.show()
fig = plt.figure ()

train['nr.employed'].plot.box(figsize=(16,5))
plt.show()


## sort cons price by education
plt.rcParams.update({'font.size': 10})
fig = plt.figure ()

train.boxplot(column='nr.employed', by = 'education')
plt.suptitle('')

fig = plt.figure ()

train.boxplot(column='nr.employed', by = 'y')
plt.suptitle('')


# Categorical Independent Variable vs Target Variable
plt.rcParams.update({'font.size': 14})

Edcucation=pd.crosstab(train['education'],train['y'])
Edcucation.div(Edcucation.sum(1).astype(float), axis=0).plot(kind='bar',stacked=True,figsize=(4,4))
plt.tight_layout()

plt.show()

marital=pd.crosstab(train['marital'],train['y'])
marital.div(marital.sum(1).astype(float), axis=0).plot(kind='bar',stacked=True,figsize=(4,4))
plt.tight_layout()
plt.show()



# Numerical  Independent Variable vs Target Variable

marital=pd.crosstab(train['emp.var.rate'],train['y'])
marital.div(marital.sum(1).astype(float), axis=0).plot(kind='bar',stacked=True,figsize=(4,4))
plt.tight_layout()
plt.show()


train.groupby('y')['emp.var.rate'].mean().plot.bar()

plt.title('y vs emp.var.rate')

# Now let’s look at the correlation between all the numerical variables.
plt.rcParams.update({'font.size': 10})

matrix = train.corr()
f, ax = plt.subplots(figsize=(9,6))
sns.heatmap(matrix,vmax=.8,square=True,cmap='BuPu', annot = True)

# Missing value imputation

train.isnull().sum()

## non which is good but we still have unkowns



# Model A logistic regression

X = train.drop('y',1)
y = train.y

X = pd.get_dummies(X)
train=pd.get_dummies(train)


from sklearn.model_selection import train_test_split
x_train, x_cv, y_train, y_cv = train_test_split(X,y, test_size=0.3)

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
model = LogisticRegression()
model.fit(x_train, y_train)
LogisticRegression()
pred_cv = model.predict(x_cv)
accuracy_score(y_cv,pred_cv)


X_test = test.drop('y',1)
y_test = test.y
X_test= pd.get_dummies(X_test)

test=pd.get_dummies(test)


x_train, x_cv, y_train, y_cv = train_test_split(X,y, test_size=0.3)

pred_test = model.predict(X_test)
accuracy_score(pred_test,y_test)


# Logistic Regression using stratified k-folds cross-validation

from sklearn.model_selection import StratifiedKFold
train = bank_dataset.iloc[:32950,:]
X = train.drop('y',1)
y = train.y

X = pd.get_dummies(X)
train=pd.get_dummies(train)

test = bank_dataset.iloc[32950:,:]

test = test.drop('y',1)

test=pd.get_dummies(test)


i=1
mean = 0
kf = StratifiedKFold(n_splits=100,random_state=1)
for train_index,test_index in kf.split(X,y):
 print ('\n{} of kfold {} '.format(i,kf.n_splits))
 xtr,xvl = X.loc[train_index],X.loc[test_index]
 ytr,yvl = y[train_index],y[test_index]
 model = LogisticRegression(random_state=1)
 model.fit(xtr,ytr)
 pred_test=model.predict(xvl)
 score=accuracy_score(yvl,pred_test)
 mean += score
 print ('accuracy_score',score)
 i+=1
 pred_test = model.predict(test)
 pred = model.predict_proba(xvl)[:,1]
print ('\n Mean Validation Accuracy',mean/(i-1))



test = bank_dataset.iloc[32950:,:]

X_test = test.drop('y',1)
y_test = test.y
X_test= pd.get_dummies(X_test)

test=pd.get_dummies(test)


x_train, x_cv, y_train, y_cv = train_test_split(X,y, test_size=0.3)

pred_test = model.predict(X_test)
accuracy_score(pred_test,y_test)

 ## ROC curve

from sklearn import metrics
y_testt = pd.get_dummies(y_test)
pred_testt = pd.get_dummies(pred_test)

fpr, tpr, _ = metrics.roc_curve(y_testt.yes, pred_testt.yes)

auc = metrics.roc_auc_score(y_testt.yes, pred_testt.yes)
plt.figure(figsize=(12,8))
plt.plot(fpr, tpr, label='validation, auc='+str(auc))
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend(loc=4)
plt.show()




from sklearn import metrics
y_testt = pd.get_dummies(y_train)
pred_test = model.predict(x_train)

pred_testt = pd.get_dummies(pred_test)

fpr, tpr, _ = metrics.roc_curve(y_testt.yes, pred_testt.yes)

auc = metrics.roc_auc_score(y_testt.yes, pred_testt.yes)
plt.figure(figsize=(12,8))
plt.plot(fpr, tpr, label='validation, auc='+str(auc))
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend(loc=4)
plt.show()


# Boosting XGBOOST

from sklearn.model_selection import StratifiedKFold
train = bank_dataset.iloc[:41188,:]
X = train.drop('y',1)
y = train.y

X = pd.get_dummies(X)
train=pd.get_dummies(train)

test = bank_dataset.iloc[10000:,:]

test = test.drop('y',1)

test=pd.get_dummies(test)
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier
i=1
mean = 0
kf = StratifiedKFold(n_splits=5,random_state=1,shuffle=True)
for train_index,test_index in kf.split(X,y):
    print ('\n{} of kfold {} '.format(i,kf.n_splits))
    xtr,xvl = X.loc[train_index],X.loc[test_index]
    ytr,yvl = y[train_index],y[test_index]
    model = XGBClassifier(n_estimators=120, max_depth=30)
    model.fit(xtr,ytr)
    pred_test=model.predict(xvl)
    score=accuracy_score(yvl,pred_test)
    mean += score
    print ('accuracy_score',score)
    i+=1
    pred_test = model.predict(X)
    pred = model.predict_proba(xvl)[:,1]
print ('\n Mean Validation Accuracy',mean/(i-1))
importances=pd.Series(model.feature_importances_, index=X.columns)
importances.plot(kind='barh', figsize=(12,8))


# Identify a variable which is most predictive

# Iterative approach
train = bank_dataset.iloc[:32950, :]

train_variables = train.columns
train_variables
accuracy_drop_out = [];
for i in range(20):

    sample_drop = train_variables[i]

    train = bank_dataset.iloc[:32950,:]
    X = train.drop('y',1)
    X = X.drop(sample_drop,1)

    y = train.y

    X = pd.get_dummies(X)
    train=pd.get_dummies(train)

    x_train, x_cv, y_train, y_cv = train_test_split(X,y, test_size=0.3)

    model = LogisticRegression()
    model.fit(x_train, y_train)
    LogisticRegression()
    pred_cv = model.predict(x_cv)
    AC = accuracy_score(y_cv,pred_cv)

    accuracy_drop_out.append(AC)


# the 5 variables selected are Contact, duration, pdays, euribor3m and cons.conf.index

train = bank_dataset.iloc[:32950,:]
X = train.drop('y', 1)
X = X.drop('age', 1)
X = X.drop('job', 1)
X = X.drop('marital', 1)
X = X.drop('education', 1)
X = X.drop('default', 1)
X = X.drop('loan', 1)
X = X.drop('month', 1)
X = X.drop('day_of_week', 1)
X = X.drop('campaign', 1)
X = X.drop('previous', 1)
X = X.drop('poutcome', 1)
X = X.drop('emp.var.rate', 1)

X = X.drop('cons.price.idx', 1)
X = X.drop('cons.conf.idx', 1)
X = X.drop('housing', 1)
X = X.drop('euribor3m', 1)

y = train.y
X = pd.get_dummies(X)
train=pd.get_dummies(train)


test = bank_dataset.iloc[32950:,:]
XX = test.drop('y',1)

XX = XX.drop('age', 1)
XX = XX.drop('job', 1)
XX = XX.drop('marital', 1)
XX = XX.drop('education', 1)
XX = XX.drop('default', 1)
XX = XX.drop('loan', 1)
XX = XX.drop('month', 1)
XX = XX.drop('day_of_week', 1)
XX = XX.drop('campaign', 1)
XX = XX.drop('previous', 1)
XX = XX.drop('poutcome', 1)
XX = XX.drop('emp.var.rate', 1)

XX = XX.drop('cons.price.idx', 1)
XX = XX.drop('cons.conf.idx', 1)
XX = XX.drop('housing', 1)
XX = XX.drop('euribor3m', 1)
XX=pd.get_dummies(XX)

YY = test.y


# Boosting XGBOOST

from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier
i=1
mean = 0
kf = StratifiedKFold(n_splits=5,random_state=1,shuffle=True)
for train_index,test_index in kf.split(X,y):
    print ('\n{} of kfold {} '.format(i,kf.n_splits))
    xtr,xvl = X.loc[train_index],X.loc[test_index]
    ytr,yvl = y[train_index],y[test_index]
    model = XGBClassifier(n_estimators=120, max_depth=30)
    model.fit(xtr,ytr)
    pred_test=model.predict(xvl)
    score=accuracy_score(yvl,pred_test)
    mean += score
    print ('accuracy_score',score)
    i+=1
    pred_test = model.predict(XX)
    pred = model.predict_proba(xvl)[:,1]
print ('\n Mean Validation Accuracy',mean/(i-1))

pred_test = model.predict(XX)
accuracy_score(pred_test,YY)

# Model B

# L.R 5 variables

i=1
mean = 0
kf = StratifiedKFold(n_splits=100,random_state=1)
for train_index,test_index in kf.split(X,y):
 print ('\n{} of kfold {} '.format(i,kf.n_splits))
 xtr,xvl = X.loc[train_index],X.loc[test_index]
 ytr,yvl = y[train_index],y[test_index]
 model = LogisticRegression(random_state=1)
 model.fit(xtr,ytr)
 pred_test=model.predict(xvl)
 score=accuracy_score(yvl,pred_test)
 mean += score
 print ('accuracy_score',score)
 i+=1
 pred_test = model.predict(XX)
 pred = model.predict_proba(xvl)[:,1]
print ('\n Mean Validation Accuracy',mean/(i-1))


pred_test = model.predict(XX)
accuracy_score(pred_test,YY)


## Model B. D.T

from sklearn import tree
i=1
mean = 0
kf = StratifiedKFold(n_splits=5,random_state=1,shuffle=True)
for train_index,test_index in kf.split(X,y):
    print ('\n{} of kfold {} '.format(i,kf.n_splits))
    xtr,xvl = X.loc[train_index],X.loc[test_index]
    ytr,yvl = y[train_index],y[test_index]
    model = tree.DecisionTreeClassifier(random_state=1)
    model.fit(xtr,ytr)
    pred_test=model.predict(xvl)
    score=accuracy_score(yvl,pred_test)
    mean += score
    print ('accuracy_score',score)
    i+=1
    pred_test = model.predict(XX)
    pred = model.predict_proba(xvl)[:,1]
print ('\n Mean Validation Accuracy',mean/(i-1))

pred_test = model.predict(XX)
accuracy_score(pred_test,YY)


## R.F

from sklearn.ensemble import RandomForestClassifier
i=1
mean = 0
kf = StratifiedKFold(n_splits=5,random_state=1,shuffle=True)
for train_index,test_index in kf.split(X,y):
    print ('\n{} of kfold {} '.format(i,kf.n_splits))
    xtr,xvl = X.loc[train_index],X.loc[test_index]
    ytr,yvl = y[train_index],y[test_index]
    model = RandomForestClassifier(random_state=1, max_depth=10)
    model.fit(xtr,ytr)
    pred_test=model.predict(xvl)
    score=accuracy_score(yvl,pred_test)
    mean += score
    print ('accuracy_score',score)
    i+=1
    pred_test = model.predict(XX)
    pred = model.predict_proba(xvl)[:,1]
print ('\n Mean Validation Accuracy',mean/(i-1))

pred_test = model.predict(XX)
accuracy_score(pred_test,YY)

## R.F with grid search

from sklearn.model_selection import GridSearchCV
paramgrid = {'max_depth': list(range(1,20,2)), 'n_estimators': list(range(1,200,20))}
grid_search=GridSearchCV(RandomForestClassifier(random_state=1),paramgrid)
from sklearn.model_selection import train_test_split
x_train, x_cv, y_train, y_cv = train_test_split(X,y, test_size=0.3, random_state=1)
grid_search.fit(x_train,y_train)
GridSearchCV(estimator=RandomForestClassifier(random_state=1),
             param_grid={'max_depth': [1, 3, 5, 7, 9, 11, 13, 15, 17, 19],
                         'n_estimators': [1, 21, 41, 61, 81, 101, 121, 141, 161,
                                          181]})
grid_search.best_estimator_
RandomForestClassifier(max_depth=5, n_estimators=41, random_state=1)
i=1
mean = 0
kf = StratifiedKFold(n_splits=10,random_state=1,shuffle=True)
for train_index,test_index in kf.split(X,y):
    print ('\n{} of kfold {} '.format(i,kf.n_splits))
    xtr,xvl = X.loc[train_index],X.loc[test_index]
    ytr,yvl = y[train_index],y[test_index]
    model = RandomForestClassifier(random_state=20, max_depth=60, n_estimators=100)
    model.fit(xtr,ytr)
    pred_test = model.predict(xvl)
    score = accuracy_score(yvl,pred_test)
    mean += score
    print ('accuracy_score',score)
    i+=XX
    pred_test = model.predict(XX)
    pred = model.predict_proba(xvl)[:,1]
print ('\n Mean Validation Accuracy',mean/(i-1))

pred_test = model.predict(XX)
accuracy_score(pred_test,YY)


# Boosting XGBOOST

from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier
i=1
mean = 0
kf = StratifiedKFold(n_splits=5,random_state=1,shuffle=True)
for train_index,test_index in kf.split(X,y):
    print ('\n{} of kfold {} '.format(i,kf.n_splits))
    xtr,xvl = X.loc[train_index],X.loc[test_index]
    ytr,yvl = y[train_index],y[test_index]
    model = XGBClassifier(n_estimators=120, max_depth=30)
    model.fit(xtr,ytr)
    pred_test=model.predict(xvl)
    score=accuracy_score(yvl,pred_test)
    mean += score
    print ('accuracy_score',score)
    i+=1
    pred_test = model.predict(XX)
    pred = model.predict_proba(xvl)[:,1]
print ('\n Mean Validation Accuracy',mean/(i-1))

pred_test = model.predict(XX)
accuracy_score(pred_test,YY)



# Boosting XGBOOST + hyperspace


csv_str = 'C:\joe file/hypeerall.csv';

df_columns = ['n_estimators', 'depth','n_splits','random_state', 'acc'];
df = pd.DataFrame(columns=df_columns)
df.to_csv(csv_str, sep=',', index=False)
from xgboost import XGBClassifier

n_estimators = np.arange(50,400,20)
max_depth = np.arange(10,100,5)
n_splits = np.arange(10,50,3)
random_state = np.arange(1,15,2)

from random import randrange
print(randrange(10))

for i in range(100):
    from sklearn.ensemble import RandomForestClassifier

    est  = n_estimators[randrange(18)]
    dep  = max_depth[randrange(18)]
    split  = n_splits[randrange(14)]
    state  = n_splits[randrange(7)]

    i=1
    mean = 0
    kf = StratifiedKFold(n_splits=split,random_state=state,shuffle=True)
    for train_index,test_index in kf.split(X,y):
        print ('\n{} of kfold {} '.format(i,kf.n_splits))
        xtr,xvl = X.loc[train_index],X.loc[test_index]
        ytr,yvl = y[train_index],y[test_index]
        model = XGBClassifier(n_estimators=est, max_depth=dep)
        model.fit(xtr,ytr)
        pred_test=model.predict(xvl)
        score=accuracy_score(yvl,pred_test)
        mean += score
        print ('accuracy_score',score)
        i+=1
        pred_test = model.predict(XX)
        pred = model.predict_proba(xvl)[:,1]
    print ('\n Mean Validation Accuracy',mean/(i-1))

    pred_test = model.predict(XX)
    acc = accuracy_score(pred_test,YY)
    print(acc)

    df = pd.DataFrame([[str(est), str(dep),
                            str(split), str(state),
                            str(acc)]],
                          columns=df_columns)


    with open(csv_str, 'a') as f:
        df.to_csv(f, header=False, encoding='utf8', line_terminator='\n', index=False)


## Important function
train = bank_dataset.iloc[:32950,:]

X = train.drop('y',1)
y = train.y

X = pd.get_dummies(X)
train=pd.get_dummies(train)
test = bank_dataset.iloc[32950:,:]
XX = test.drop('y',1)
XX = pd.get_dummies(XX)


from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier
i=1
mean = 0
kf = StratifiedKFold(n_splits=5,random_state=1,shuffle=True)
for train_index,test_index in kf.split(X,y):
    print ('\n{} of kfold {} '.format(i,kf.n_splits))
    xtr,xvl = X.loc[train_index],X.loc[test_index]
    ytr,yvl = y[train_index],y[test_index]
    model = XGBClassifier(n_estimators=120, max_depth=30)
    model.fit(xtr,ytr)
    pred_test=model.predict(xvl)
    score=accuracy_score(yvl,pred_test)
    mean += score
    print ('accuracy_score',score)
    i+=1
    #pred_test = model.predict(X_test)
    pred = model.predict_proba(xvl)[:,1]
print ('\n Mean Validation Accuracy',mean/(i-1))


pred_test = model.predict(X_test)
accuracy_score(pred_test,y_test)


importances=pd.Series(model.feature_importances_, index=X.columns)
importances.plot(kind='barh', figsize=(12,8))