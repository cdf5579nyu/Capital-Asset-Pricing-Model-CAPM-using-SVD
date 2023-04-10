#!/usr/bin/env python
# coding: utf-8

# Please follow the instructions carefully. Write all your code in a `Code` cell, and your explanations in a `Markdown` cell. Make sure that your code compiles correctly either by selecting a given cell and clicking the `Run` button, or by hitting `shift`+`enter` or `shift`+`return`.

# ### 1. Import `numpy`, `numpy.linalg`, `matplotlib.pyplot`, and `pandas`. Use the appropriate aliases when importing these modules.

# In[2]:


# code for question 1
import numpy as np
import matplotlib.pyplot as plt 
import pandas as pd
import numpy.linalg as la


# ### 2. Load the data from the file named `data_stock_returns.csv` into a `DatFrame` called `returns`. The file `data_stock_returns.csv` contains daily returns of a number of stocks selected from the S&P 500 universe. The rows of the csv file represent the returns over a number of days, and the columns represent individual stocks labeled by their NYSE ticker symbol, e.g., Apple stock is labeled `AAPL`.

# In[3]:


# code for question 2
returns = pd.read_csv("data_stock_returns.csv")  # load file as 


# ### 3. View the `head` of the `returns` `DataFrame`

# In[4]:


# code for question 3
returns.head()


# ### 4. View the `tail` of the `returns` `DataFrame`

# In[5]:


# code for question 4
returns.tail()


# ### 5. How many stocks are in this `DataFrame`?

# In[6]:


# code for question 5
len(returns.T)  #just transposing the matrix 


# **ANSWER FOR QUESTION 5**: double click on this cell to write your answer: 489 different stocks

# ### 6. Over how many days are these stock returns reported?

# In[7]:


# code for question 6
len(returns)


# **ANSWER FOR QUESTION 6**: double click on this cell to write your answer: 252 trading days

# ### 7. Extract the returns of the Amazon stock only, which has a ticker symbol `AMZN`. Save it in a `Series` called `amzn_returns`.

# In[8]:


# code for question 7
amzn_returns = returns['AMZN']


# ### 8. Plot the Amazon stock returns extracted in the above cell. 

# In[9]:


# code for question 8
plt.plot(amzn_returns)
plt.title("Amazon returns on stocks")
plt.xlabel("Days registered")
plt.ylabel("Returns")
#give x and y axis def
plt.show()


# ### 9. Plot the cumulative sum of the Amazon stock returns using the method `.cumsum()` which acts directly on the `amzn_returns` `Series`.

# In[98]:


# code for question 9
plt.plot(amzn_returns.cumsum(), "g")
plt.title("Amazon cumulative returns on stocks")
plt.xlabel("Days registered")
plt.ylabel("Cummulative returns")
#give x and y axis def
plt.show()


# In[11]:


# the module below will allow us to perform linear regression
import statsmodels.api as sm


# The function `lin_reg(x,y)` given below performs ordinary least squares (OLS) linear regression using `sm.OLS` from the `statsmodels.api` module.
# 
# The code enclosed in `''' '''` is the docstring of the function `lin_reg`.
# 
# `x` in the `lin_reg` function is a matrix that contains the regressors, and `y` represents the vector containing the dependent variable. Note that `x` might contain one vector or multiple vectors. In the case that `x` contains one vector $x$, the regression gives:
# 
# $$ y = \beta_0 + \beta_1 x $$
# 
# In the case that `x` contains multiple vectors $x_1, \dots, x_k$, the regression becomes:
# 
# $$ y = \beta_0 + \beta_1 x_1 + \dots + \beta_k x_k$$
# 
# The $\beta$'s are the regression coefficients obtained using least squares. Note that `sm.add_constant` is used in the function below to make `x` look like the matrix $A$ we use in least squares, whose first column contains all ones.

# In[12]:


def lin_reg(x, y):
    '''
    oridinary linear regression using least-squares
    
    Parameters
    ----------  
    x: regressors (numpy array)
    y: dependent variable (numpy array)
    
    Returns
    -------
    coefficients: regression coefficients (pandas Series)
    residuals: regression residuals (numpy array)
    r_squared: correlation coefficient (float)
    
    '''
    x = sm.add_constant(x)
    model = sm.OLS(y, x).fit()
    coefficients = model.params
    residuals = model.resid
    r_squared = model.rsquared
    return coefficients, residuals, r_squared


# ### 10. Let's try to use the obove function. Extract (as numpy array) the stock returns of:
# 
# - Apple (ticker symbol `AAPL`) and call it `aapl`
# - Intel (ticker symbol `INTC`) and call it `intc`
# - Microsoft (ticker symbol `MSFT`) and call it `msft`
# - IBM (ticker symbol `IBM`) and call it `ibm`
# 
# ### Let `y` be the Apple stock returns, and `x` be the Intel stock returns. Use the `lin_reg` function defined above to find $y=\beta_0 + \beta_1 x$. 

# In[29]:


# code for question 10
aapl =  returns['AAPL'].to_numpy()
intc =  returns['INTC'].to_numpy()
msft =  returns['MSFT'].to_numpy()
ibm =  returns['IBM'].to_numpy()

y = aapl
x = intc
coef, residuals, r_square = lin_reg(x,y)
print(r_square)
func = 0.00195633 + 0.53526326*x #this will represent out linear model y hat
y_hat = coef[0] + coef[1]*x #this is an equivalent method


# ### 11. Plot the cumulative sum of the Apple returns prediction from least squares on top of the actual Apple returns. How well do the Intel stock returns describe the Apple stock returns?

# In[104]:


# code for question 11
aaplpred = func.cumsum()

plt.plot(aapl.cumsum(), label="Apple cumulative returns") #cumulative sum of apple returns

plt.plot(aaplpred, "-r", label="Predicted cumulative returns") #setting up the prediction shortfalls from our regression
plt.title("Apple returns on stocks vs predictive model using intel")
plt.figtext(.68, .2, "r_square = 0.3050")
plt.xlabel("Days registered")
plt.ylabel("Cummulative returns")
plt.legend(loc="upper left")
#give x and y axis def
plt.show()

#the red line is our prediction


# In[97]:


# code for question 11 but using a different method
aaplpred = (aapl-residuals).cumsum() #by using residuals we could apply the same
plt.plot(aapl.cumsum()) #cumulative sum of apple returns 
plt.plot(aaplpred, "-r") #setting up the prediction shortfalls from our regression 
plt.title("Apple returns on stocks vs predictive model using intel") 
plt.xlabel("Days registered") 
plt.ylabel("Cummulative returns") #give x and y axis def 
plt.show() #the green line is our prediction


# So as we can see, the model seems to be touching the cumulative sum of the apple returns, but the trend doesn't seem to be perfectly reflected by our model, and that is probably why the r_square is not that high in our model, seems to be overshooting and undershooting on specific behaviors. However, given that it comes from only one stock, the model is pretty good in those standarts.

# ### 12. Now, let `y` be the Apple stock returns, and `x` be the Intel, Microsoft, and IBM stock returns. Use the `lin_reg` function defined above to find $y=\beta_0 + \beta_1 x_1 + \beta_2 x_2 + \beta_3 x_3$, where $x_1$ represents Intel returns, $x_2$ represents Microsoft returns, and $x_3$ represents IBM returns. 

# In[35]:


# code for question 12
y = aapl
x1 = intc
x2 = msft
x3 = ibm
x4 = returns[['INTC', 'MSFT','IBM']]
coef1, residuals1, r_square1 = lin_reg(x4,y)
print(coef1)
func_multi = 0.001116 + 0.284141*x1 + 0.543723*x2 + 0.150586*x3
y_hat2 = coef1[0] + coef1[1]*x1 + coef1[2]*x2 + coef1[3]*x3 #is an equivalent form of the coefficients


# ### 13. Plot the cumulative sum of the Apple returns prediction from least squares on top of the cumulative sum of actual Apple returns. How well do the Intel, Microsoft, and IBM stock returns describe the Apple stock returns?

# In[103]:


# code for question 13
plt.plot(aapl.cumsum(), label="Apple cumulative returns") #cumulative sum of apple returns
#plt.plot(intc.cumsum()) #cumulative sum of apple returns

plt.plot(func_multi.cumsum(), "g",label="Predicted cumulative returns") #setting up the prediction shortfalls from our regression
plt.title("Apple returns on stocks vs Predictive multivariable model")

plt.figtext(.66, .2, "r_square = 0.47047")

plt.xlabel("Days registered")
plt.ylabel("Cummulative returns")
plt.legend(loc="upper left")

#give x and y axis def
plt.show()


# Now, we can see that our model is doing a better job in predicting the trends of when the cumulative returns of apple will go up and down. We can see that the main issue is that it seems to be undershooting, but the variability of the trends is being almost perfectly replicated. As we can see with regards to the r_square, the model fits our data better than the last one that dependent in one variable. However, i did not significally change by adding these new variables, and diminishes principles of simplification over complexity of forecasting. To improve this model, we could also probably add a constant to drive the whole graph up.

# The file `SPY.csv` contains the prices of SPDR S&P 500 ETF Trust. This Exchage Traded Fund (ETF) contains a collection of assets currently present in the S&P 500 index. 
# 
# ### 14. Load `SPY.csv` into a DataFrame called `spy_prices` using the `read_csv` method in `pandas`. Make sure to make the 'Date' column to be your index column. To do that, read the docstring for `read_csv`. 

# In[40]:


# code for question 14
spy_prices = pd.read_csv("SPY.csv", index_col=0) #we load the dataset and set the date column as index colum 


# ### 15. Once you have downloaded the file into the `DataFrame`, observe all the available prices and dates. Show the head of the `DataFrame`, and then answer the following questions:
# 
# (a) Which prices are reported?
# 
# (b) From which date to which date are these prices reported?

# In[41]:


# code for question 15
spy_prices.head()


# In[46]:


spy_prices.tail()


# **ANSWER FOR QUESTION 15**: double click on this cell to write your answer
# 
# (a) Which prices are reported?: for each day, we have the highest and the lowest value it had, the value the stock had at the opening of the markets, the price in which it closed the trading day, then the volume of stocks traded and the adj close which takes the closing value and adjustes it after accounting for any corporate actions
# 
# (b) From which date to which date are these prices reported?: from 2015-01-02 (January 2nd, 2015) to 2020-06-01 (June 6th, 2020)

# ### 16. Retain only the Adjusted Close price in the `spy_prices` `DataFrame`. Call the resulting `Series` `spy_adjclose`.

# In[47]:


# code for question 16
spy_adjclose = spy_prices["Adj Close"]
print(spy_adjclose)


# ### 17. Now, using the `pct_change` method in `pandas`, compute the returns on the Adjusted Close prices of SPY, and only retain the returns from '2019-01-01' to '2020-01-01'. Call the `Series` obtained `spy_returns`.

# In[48]:


# code for question 17
spy_returns = (spy_adjclose["2019-01-01":"2020-01-01"].pct_change()).fillna(0)

#to handle the first nan value and for the computation to work, we could either:
#find the percent change from december 31st 2018 to put it in our first location or
#set the first value equal to zero. Since I check that there are no other nan values on 
#the given dataset, so putting fillna won't damage our model in the future.

#Moreover, the value of the first percent change is close to zero as well, the it won't
#dramatically change the outcomes 

print(spy_returns)


# ### 18. Perform SVD on `returns` data that contain assets from the S&P 500. Retain the left singular vector corresponding to the largest singular value and call is `u_sigma1`.

# In[49]:


# code for question 18
A = returns.drop(['Date'], axis=1).to_numpy() #converted into a matrix to use svd, and numpy series

#print(A)
u, s, vh = la.svd(A, full_matrices=True) # u, sigma and v transposed 

sh = np.diag(s, k=0)
print(A.shape)
print(u.shape)
print(sh.shape)
print(vh.shape)

#print(np.dot(a = (np.dot(a = u, b = sh)), b = vh))
#sh only has the diagonal values but its missing zeros in its diagonal to acquire 252x488 dimensions


# ### 19. `u_sigma1` is thought to track the market. To test that, we will perform a regression of `spy_returns` against this first left singular vector by letting `y=spy_returns` and `x=u_sigma1` and computing
# 
# ### $$ y = \beta_0 + \beta_1 x$$
# ### using least squares regression.

# In[81]:


# code for question 19
x_for_u = u.T[0] #the vectors of u related to sigma one (transpose cause of the setup of columns in python)

y = spy_returns[0:].to_numpy() #we set it up as a numpy array to do the linear regression

coef3, residuals3, r_square3 = lin_reg(x,y)
print(r_square3)

func3 = 0.00034244 - 0.1189713*x_for_u
y_hat3= coef3[0] + coef3[1]*x_for_u #a similar function than the one on top


# ### 20. Plot the cumulative sum of the result from the regression on top of the cumulative sum of `spy_returns`. What do you notice?

# In[94]:


# code for question 20
aaplpred = func3.cumsum()

plt.plot(spy_returns.cumsum(), label="Spy cumulative returns") #cumulative sum of apple returns

plt.plot(aaplpred, label="Predicted cumulative returns using U-sigma1") #setting up the prediction shortfalls from our regression
plt.title("Spy cumulative returns on stocks vs predictive model using svd on 2019")
plt.xlabel("Days registered")
plt.ylabel("Cumulative returns")
plt.legend(loc="upper left")

plt.xticks(np.arange(spy_returns[0], len(spy_returns), 80))

plt.figtext(.66, .2, "r_square = 0.8984")
#give x and y axis def
plt.show()


# **ANSWER FOR QUESTION 20**: double click on this cell to write your answer:
# Now we can see that the model using svd is pretty accurate when predicting the cumulative returns of spy, and at some points you cannot even tell the difference between which one is which. As a result of that, we see that the r_square is dramatically higher than our past models. And the reason behind this model fitting so well with the data is because the singular value decomposition used for the linear model is coming from the same dataset but only using sigma 1. This means that if we use the other sigmas, we are probably going to get the exact same line. And thats the importance of svd, in oversimplifying our issue so that we don't need to much data to predict the movements of certain cumulative returns.

# # Congratulations! You have just implemented your first statistical Capital Asset Pricing Model (CAPM) to the S&P 500 market.
