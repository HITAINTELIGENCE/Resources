import warnings
from sklearn.datasets import load_breast_cancer
import numpy as np

breast_cancer = load_breast_cancer()
X = breast_cancer.data
y = breast_cancer.target

print(y)

# y = [1 if price > 20 else 0 for price in y] # 1 nhà giá cao, 0 nhà giá thấp
# y = np.array(y)

# print('Input features: ', ', '.join(breast_cancer.feature_names))
# print(X.shape)