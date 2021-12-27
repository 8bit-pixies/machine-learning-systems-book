---
title: Transfer Learning with Embeddings
---

Another approach is to use transfer learning for build machine learning models. In low resource settings, we typically do not want to go down the fine tuning route. Its difficult to get right and not guarenteed to get the results we want. Instead, we can use the embedding layers as inputs for another machine learning model. One simple approach is to use approximate nearest neighbours over the pre-built embeddings as the basis to infer concepts to a new domain. 

Variation: using `annoy`

```py
from sklearn.base import ClassifierMixin
from sklearn.neighbors import KNeighborsClassifier
from annoy import AnnoyIndex  # pip install annoy
import pandas as pd
import numpy as np
from sklearn import datasets


class FastANN(ClassifierMixin):
    def __init__(self, n_neighbors=5, n_trees=100, metric='angular'):
        self.n_neighbors = n_neighbors
        self.n_trees = n_trees
        self.metric = metric

    def fit(self, X, y):
        self.y = y
        self.ss = pd.Series(y).value_counts() * 0
        self.ann = AnnoyIndex(X.shape[1], self.metric)
        for i in range(X.shape[0]):
            self.ann.add_item(i, X[i, :])
        
        self.ann.build(self.n_trees)
        return self
    
    def predict_single_proba(self, v):
        indx = self.ann.get_nns_by_vector(v, self.n_neighbors)
        ss = (self.ss + pd.Series(y[indx]).value_counts(normalize=True)).fillna(0).values
        return ss

    def predict_(self, X):
        pred = []
        for i in range(X.shape[0]):
            pred.append(self.predict_single_proba(X[i, :]))
        return np.stack(pred, 0)
    
    def predict_proba(self, X):
        return self.predict_(X)
    
    def predict(self, X):
        return np.argmax(self.predict_(X), axis=1)

X, y = datasets.load_breast_cancer(return_X_y = True)

model = FastANN(n_neighbors=5, n_trees=10)
model.fit(X, y)
model.score(X, y)

model2 = KNeighborsClassifier()
model2.fit(X, y)
model2.score(X, y)
```

Variation: using `faiss`

```py
from sklearn.base import ClassifierMixin
from sklearn.neighbors import KNeighborsClassifier
import faiss  # conda install -c pytorch faiss-cpu
import pandas as pd
import numpy as np
from sklearn import datasets


class FastANN(ClassifierMixin):
    def __init__(self, n_neighbors=5, n_trees=100, metric='angular'):
        self.n_neighbors = n_neighbors
        self.n_trees = n_trees
        self.metric = metric

    def fit(self, X, y):
        X = X.astype(np.float32)
        self.y = y
        self.ss = pd.Series(y).value_counts() * 0
        self.ann = faiss.IndexFlatL2(X.shape[1])
        self.ann.add(X)
       
        return self
    
    def predict_single_proba(self, v):
        _, indx = self.ann.search(v, self.n_neighbors)
        ss = (self.ss + pd.Series(y[indx.flatten()]).value_counts(normalize=True)).fillna(0).values
        return ss

    def predict_(self, X):
        X = X.astype(np.float32)
        pred = []
        for i in range(X.shape[0]):
            pred.append(self.predict_single_proba(X[[i], :]))
        return np.stack(pred, 0)
    
    def predict_proba(self, X):
        return self.predict_(X)
    
    def predict(self, X):
        return np.argmax(self.predict_(X), axis=1)

X, y = datasets.load_breast_cancer(return_X_y = True)

model = FastANN(n_neighbors=5, n_trees=10)
model.fit(X, y)
model.score(X, y)

model2 = KNeighborsClassifier()
model2.fit(X, y)
model2.score(X, y)
```

Although the performance in both variations is worse than using the built-in k-nearest neighbors classifier, it is an order of magnitude faster on larger datasets. This kind of fine-tuning can 