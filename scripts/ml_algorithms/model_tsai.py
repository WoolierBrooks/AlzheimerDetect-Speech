from tsai.all import *
import sklearn.metrics as skm

print(get_UCR_univariate_list())
print(get_UCR_multivariate_list())


# dataset id
dsid = 'NATOPS' 
X, y, splits = get_UCR_data(dsid, return_split=False)

print(f"dsid: {dsid}")
print(f"X: {X}")
print(f"y: {y}")
print(f"splits: {splits}")

X.shape, y.shape, splits