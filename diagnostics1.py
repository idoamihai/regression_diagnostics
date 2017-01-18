# -*- coding: utf-8 -*-
"""
Created on Thu Jan 12 10:54:22 2017

@author: idoamihai
"""
import numpy as np
import scipy
import pandas as pd
import matplotlib.pyplot as plt

#stats
def get_stats(clf,x,y,regression):
    '''
    Generates a pandas dataframe with regression statistics.
    clf is a classifier with a predict function (e.g. an sklearn classifier).
    x and y are the predictor and predicted variables, respectively.
    regression can be True or False dependent on whether we're working
    with a regression or classification problem.
    '''
    if regression:
        predProbs = np.matrix(clf.predict(x))
    else:
        predProbs = np.matrix(clf.predict_proba(x))
    # Design matrix -- add column of 1's at the beginning of your X_train matrix
    if "intercept_" in dir(clf):
        intercept = clf.intercept_
    else:
        intercept = clf.named_steps['clf'].intercept_
    X_design = np.hstack((np.ones(shape = [x.shape[0],1])*intercept, x))
    if regression:
        MSE = np.mean((y-predProbs.A1)**2)
        var = MSE * np.diag(np.linalg.pinv(np.dot(X_design.T,X_design)))
        se = np.sqrt(var)
    else:
        # Initiate matrix of 0's, fill diagonal with each predicted observation's variance
        V = np.matrix(np.zeros(shape = (X_design.shape[0], X_design.shape[0])))
        np.fill_diagonal(V, np.multiply(predProbs[:,0], predProbs[:,1]).A1)
        # Covariance matrix
        covLogit = np.linalg.inv(X_design.T * V * X_design)
        se = np.sqrt(np.diag(covLogit))
    if "coef_" in dir(clf):
        coefs = np.insert(clf.coef_, 0, clf.intercept_)
    else:
        coefs = np.insert(clf.named_steps['clf'].coef_, 0, 
                          clf.named_steps['clf'].intercept_)
    statistic = coefs/se
    p_values = scipy.stats.norm.sf(abs(statistic))*2 #twosided
    features = np.r_[['intercept'],x.columns]
    stats = pd.DataFrame(np.c_[features,coefs,se,statistic,p_values],
                         columns=['feature','coef','SE','t','pval'])
    return stats

def normality_tests(x,feature):
    '''
    creates a QQ plot against the normal distribution and 
    runs a kolmogorov-smirnov statistical test.
    x is a pandas dataframe with 'feature' as a column.
    '''
    scipy.stats.probplot(x[feature].values,dist='norm',plot=plt)
    plt.title(feature)
    ks = scipy.stats.kstest(x[feature],cdf='norm')
    plt.show()
    return ks
           
def residuals_plot(clf,x,y):  
    '''
    ordinary residuals plot.
    clf is a classifier with a predict function (e.g. an sklearn classifier).
    x and y are the predictor and predicted variables, respectively.
    '''
    preds = clf.predict(x)
    residuals = y - preds
    fig = plt.figure('residuals plot')
    ax = fig.add_subplot(111)
    ax.scatter(preds,residuals)
    ax.hlines(0,xmin=ax.get_xlim()[0],xmax=ax.get_xlim()[1])
    ax.set_xlabel('fitted values')
    ax.set_ylabel('residuals') 
    plt.show()
    return preds,residuals       
  
def regression_diagnostics(clf,x,y):
    '''
    detection of outliers, high-leverage, and high-influence datapoints.
    clf is a classifier with a predict function (e.g. an sklearn classifier).
    x and y are the predictor and predicted variables, respectively.
    Outputs the hat-matrix, studentized-residuals, leverage, and cook's distance values.
    '''
    preds = clf.predict(x)
    resids = preds - y
    studentized_resids = np.zeros(y.shape[0], dtype='float')
    # Calcluate hat matrix of X values so you can get leverage scores
    hat_matrix = np.dot(
        np.dot(x, np.linalg.inv(np.dot(np.transpose(x), x))),
        np.transpose(x))
    leverage = np.diagonal(hat_matrix).copy()
    # For each point, calculate studentized residuals w/ leave-one-out MSE
    for i in range(y.shape[0]):
        # Make a mask so you can calculate leave-one-out MSE
        mask = np.ones(y.shape[0], dtype='bool')
        mask[i] = 0
        loo_mse = np.average(resids[mask] ** 2, axis=0)  # Leave-one-out MSE
        # Calculate studentized residuals
        studentized_resids[i] = resids[i] / np.sqrt(
            loo_mse * (1 - hat_matrix[i, i]))
    stand_resids =  resids/np.std(resids) #standardized residuals   
    cooks_d2 = stand_resids**2 / x.shape[1] + 1
    cooks_d2 *= leverage / (1 - leverage)
    deg_f = x.shape[0]-x.shape[1]-2
    alpha = 0.05/2/x.shape[1]
    thresh = scipy.stats.t.ppf(alpha,deg_f)
    fig = plt.figure('diagnostics plots')
    ax = fig.add_subplot(221)
    ax.scatter(preds,studentized_resids)
    ax.hlines(abs(thresh),xmin=ax.get_xlim()[0],xmax=ax.get_xlim()[1],linestyle='--')
    ax.hlines(-abs(thresh),xmin=ax.get_xlim()[0],xmax=ax.get_xlim()[1],linestyle='--')
    ax.set_xlabel('fitted values')
    ax.set_ylabel('studentized residuals')
    ax.set_title('studentized residuals plot')
    ax2 = fig.add_subplot(222)
    ax2.scatter(preds,leverage)
    ax2.hlines(leverage.mean()*2,xmin=ax2.get_xlim()[0],
               xmax=ax2.get_xlim()[1],linestyle='--')
    ax2.set_xlabel('fitted values')
    ax2.set_ylabel('leverage statistic')
    ax2.set_title('leverage plot')
    ax3 = fig.add_subplot(223)
    scipy.stats.probplot(studentized_resids,dist=scipy.stats.t(x.shape[0],deg_f),plot=plt)
    ax3.set_title('studentized_residuals vs t-distribution')
    ax4 = fig.add_subplot(224)
    ax4.scatter(preds,cooks_d2) 
    ax4.set_xlabel('fitted values')
    ax4.set_ylabel('cook''s distance')
    ax4.hlines(4./(x.shape[0]-x.shape[1]-1),xmin=ax.get_xlim()[0],xmax=ax.get_xlim()[1],linestyle='--')
    ax4.set_title('influence')
    plt.show()
    fig2 = plt.figure('influence')
    ax5 = fig2.add_subplot(111)
    ax5.scatter(preds,studentized_resids,s=cooks_d2*5000)
    ax5.set_xlabel('fitted values')
    ax5.set_ylabel('studentized residuals')
    ax5.set_title('influence (larger bubble = more influence)')
    plt.show()
    return hat_matrix, studentized_resids,leverage,cooks_d2

def logit_plot(clf,x,feature):
    '''
    Logits and probability plots.
    clf is a classifier with a predict function (e.g. an sklearn classifier).
    x and y are the predictor and predicted variables, respectively.
    '''
    idx_ = [idx for idx,val in enumerate(x.columns) if val in feature ]
    if 'named_steps' in dir(clf):
        scaler = clf.named_steps['scale']
        coefs_ = clf.named_steps['clf'].coef_[0]
        intercept = clf.named_steps['clf'].intercept_
        scaled_feature = scaler.transform(x)[:,idx_]
    else:
        coefs_ = clf.coef_[0]
        intercept = clf.intercept_
        scaled_feature = x[feature] #not scaled here but kept the variable name
    logodds_feature = intercept+(coefs_[idx_]*scaled_feature)
    odds_feature = np.exp(logodds_feature)
    prob_feature = odds_feature/(1+odds_feature)
    sort_idx = scaled_feature.ravel().argsort()
    fig = plt.figure('%s logodds plot' %feature)
    ax = fig.add_subplot(121)
    ax.plot(scaled_feature[sort_idx],logodds_feature[sort_idx],'-o')
    ax.set_xlabel('%s (scaled)'%feature)
    ax.set_ylabel('log-odds')
    ax.set_title('logits')
    ax2 = fig.add_subplot(122)
    ax2.plot(scaled_feature[sort_idx],prob_feature[sort_idx],'-o')
    ax2.set_xlabel('%s (scaled)'%feature)
    ax2.set_ylabel('probability')
    ax2.set_title('probabilities')
    return logodds_feature,prob_feature

def variance_inflation_factor(clf,x):
    vif_all = []
    for var_idx in range(x.shape[1]):
        k_vars = x.shape[1]
        x_i = x.iloc[:, var_idx]
        mask = np.arange(k_vars) != var_idx
        x_noti = x.iloc[:, mask]
        clf.fit(x_noti,x_i)   
        SST = ((x_i-x_i.mean())**2).sum()
        SSReg = ((clf.predict(x_noti)-x_i.mean())**2).sum()
        rsquared_i = SSReg/SST
        vif = 1. / (1. - rsquared_i)
        vif_all.append(vif)
    vif = pd.DataFrame(np.c_[x.columns,vif_all],
                       columns=['feature','vif'])
    return vif

def correlation_matrix(df,vars_):    
    corrs = np.corrcoef(df[vars_],rowvar=0)
    corrs = pd.DataFrame(corrs,columns=vars_)
    corrs.index = corrs.columns
    corrs = corrs.round(2)
    return corrs
