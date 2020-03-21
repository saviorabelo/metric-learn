from metric_learn import ITML_Supervised
from sklearn.datasets import load_iris

iris_data = load_iris()
X = iris_data['data']
Y = iris_data['target']

itml = ITML_Supervised(num_constraints=200)
itml.fit(X, Y)