---
title: Finding Rules from Discretized Buckets
---

The first stage in feature engineering is to determine a set of optimal rules. The typical workflow uses techniques such as:

*  Finding correlations
*  Performing EDA
*  Bucketing values
*  Determining features which are highly correlated

# Building Discretized Features

Using `scikit-learn` we can build discretized features quickly and effectively. 

```py
from sklearn.model_selection import train_test_split
from sklearn import datasets
from sklearn.preprocessing import KBinsDiscretizer


X, y = datasets.load_diabetes(return_X_y = True)
seed = 1
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=seed)

kbins = KBinsDiscretizer(n_bins=20)
kbins.fit(X_train, y_train)
output = kbins.transform(X_train)  # shape: (None, 169)
```


# Optimal using Feature Selection

Even in trivial datasets, the number of features increases exponentially, we can aggressively reduce the feature space, even when taking into account. A common approach is to expand the feature space and add a feature selection component in a pipeline. 

```py
from sklearn.model_selection import train_test_split
from sklearn import datasets, svm
from sklearn.preprocessing import KBinsDiscretizer, PolynomialFeatures
from sklearn.pipeline import make_pipeline
from sklearn.feature_selection import SelectFromModel


X, y = datasets.load_diabetes(return_X_y = True)
seed = 1
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=seed)

feature_pipeline = make_pipeline(
    PolynomialFeatures(degree=2, interaction_only=True),
    KBinsDiscretizer(n_bins=256, strategy='uniform'),
    SelectFromModel(svm.LinearSVC(penalty="l1", dual=False), max_features=512)  # without this we'll have over 2,000 features
)
feature_pipeline.fit(X_train, y_train)
output = feature_pipeline.transform(X_train)  # shape: (None, 512)
```

This approach can learn to select features based on linear assumptions. For more background on feature selection in the online setting, you can read more on Grafting. 


# (Bonus) Optimal Discretized Buckets using RuleFit

Approaches such as RuleFit will automatically find candidate features as part of its model building process. As it automatically determines interactions and optimises the feature space. To begin we need to install the interpret package which includes wrappers for RuleFit and the official implementation for Explainable Boosting Machines. 

```py
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.base import TransformerMixin
import numpy as np

from interpret.glassbox import ExplainableBoostingRegressor
from interpret import show

X, y = datasets.load_diabetes(return_X_y=True)
seed = 1
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=seed
)

ebm = ExplainableBoostingRegressor(random_state=seed)
ebm.fit(X_train, y_train)
ebm_global = ebm.explain_global()
graph_data = ebm_global.data(0)  # contains information on the feature and confidence interval
# for features which use interactions, it looks in all subsets and performs a table lookup (not implemented)


class BoostedRegressorBinsDiscretizer(TransformerMixin):
    def __init__(self, merge_bins=True, ebm_config={}):
        self.ebm_config = ebm_config
        self.ebm = None
        self.merge_bins = merge_bins

    def fit(self, X, y, **kwargs):
        # ..omitted..

    def transform(self, X):
        # ..omitted..


bbd = BoostedRegressorBinsDiscretizer()
bbd.fit(X_train, y_train)
bbd.transform(X_train)

```

This constructs a "smarter" binning approach than doing it unsupervised (or is a bit bitter than performing it unsupervised followed by feature selection). By performing a table-based lookup when considering interaction terms is the same as constructing a learned embedding over a particular space. This is an efficient way (albeit offline) construction which can help determine non-linearities for used with a linear machine learning model (e.g. as used in Explainable Boosting Machines). 

Should we use this code for production? Personally, a custom transformers which wraps around an existing model is a bit risky, as having custom code means needing to maintain custom packages. Perhaps building custom pipelines is preferable rather than ephemeral wrappers which are not officially supported. Building pipelines with appropriate metadata interfaces is probably preferable in the longer term. An improved interface could be:

```py
import pandas as pd
import numpy as np


def build_config(model):
    """
    See reference: https://github.com/interpretml/ebm2onnx/blob/master/ebm2onnx/convert.py
    """
    model_config = []
    model_global = model.explain_global()

    for feature_index in range(len(model.feature_names)):
        feature_name = model.feature_names[feature_index]
        feature_type = model.feature_types[feature_index]
        feature_group = model.feature_groups_[feature_index]
        info_config = {}

        if feature_type == "continuous":
            info_config["feature_type"] = feature_type
            info_config["column_name"] = [feature_name]
            info_config["column_index"] = [feature_group[0]]
            info_config["column_mapping"] = (
                [-np.inf]
                + list(model.preprocessor_.col_bin_edges_[feature_group[0]])
                + [np.inf]
            )
            info_config["scores"] = model.additive_terms_[feature_index][1:]
            # print(len(list(pd.IntervalIndex.from_tuples(list(zip(info_config["column_mapping"][0][:-1], info_config["column_mapping"][0][1:]))))),
            # len(info_config['scores']))
            info_config["table"] = pd.DataFrame(
                {
                    "interval": list(
                        pd.IntervalIndex.from_tuples(
                            list(
                                zip(
                                    info_config["column_mapping"][0][:-1],
                                    info_config["column_mapping"][0][1:],
                                )
                            )
                        )
                    ),
                    "scores": info_config["scores"],
                }
            )
        elif feature_type == "categorical":
            info_config["feature_type"] = feature_type
            info_config["column_name"] = [feature_name]
            info_config["column_index"] = [feature_group[0]]
            info_config["column_mapping"] = model.preprocessor_.col_mapping_[
                feature_group[0]
            ]
            info_config["scores"] = model.additive_terms_[feature_index]
            row_index = list(info_config["column_mapping"].keys())
            dummy_index = " "
            while dummy_index in row_index:
                dummy_index += " "
            row_index = [dummy_index] + row_index
            info_config["table"] = pd.DataFrame(
                {"categories": row_index, "scores": info_config["scores"]}
            )
        elif feature_type == "interaction":
            # left part right part? I think using range is harder to read - maybe.
            info_config["feature_type"] = [
                model.feature_types[idx] for idx in feature_group
            ]
            info_config["column_name"] = [
                model.preprocessor_.feature_names[idx] for idx in feature_group
            ]
            info_config["column_index"] = list(feature_group)

            if info_config["feature_type"][0] == "continuous":
                left_mapping = (
                    [-np.inf]
                    + model.pair_preprocessor_.col_bin_edges_[feature_group[0]].tolist()
                    + [np.inf]
                )
                row_index = list(
                    pd.IntervalIndex.from_tuples(
                        list(zip(left_mapping[:-1], left_mapping[1:]))
                    )
                )
            else:
                left_mapping = model.preprocessor_.col_mapping_[feature_group[0]]
                row_index = list(left_mapping.keys())
            if info_config["feature_type"][1] == "continuous":
                right_mapping = (
                    [-np.inf]
                    + model.pair_preprocessor_.col_bin_edges_[feature_group[1]].tolist()
                    + [np.inf]
                )
                col_index = list(
                    pd.IntervalIndex.from_tuples(
                        list(zip(right_mapping[:-1], right_mapping[1:]))
                    )
                )
            else:
                right_mapping = model.preprocessor_.col_mapping_[feature_group[1]]
                col_index = list(right_mapping.keys())

            info_config["column_mapping"] = [left_mapping, right_mapping]
            info_config["scores"] = model_global.data(feature_index)["scores"]
            info_config["table"] = pd.DataFrame(
                info_config["scores"], columns=col_index, index=row_index
            )
        else:
            raise ValueError(f"feature type {feature_type} is not supported.")

        model_config.append(info_config)
    return model_config


def bin_mapper(X, config):
    """
    Builds bins in a generic way, a wrapper around pandas cut with some
    extensions - supports table lookups. If mapping is not provided, 
    will be converted to a one-hot encoded version.

    bins is a list of bins, columns is a list of columns...

    Usage:

    bins = [
        [1,2,3],
        [3,4,5]
    ]

    columns = [0, 1]
    mapping = np.arange(16).reshape(4,4)
    bin_mapper(np.arange(20).reshape(10, 2), bins, columns, mapping)
    """
    output = []
    if type(X) is pd.DataFrame:
        X_ = X[config['column_name']]
    else:
        X_ = X[:, config['column_index']]
    
    if config['feature_type'] == 'continuous':
        out_ = pd.cut(X_, config['])
        
    elif config['feature_type'] == 'categorical':
    else:
    for b, c in zip(bins, columns):
        X_ = X[:, c]
        intervals = [-np.inf] + b + [np.inf] 
        out_ = pd.cut(X_, intervals, labels=False)
        if mapping is None:
            out_ = pd.get_dummies(out_)
        output.append(out_)

    if mapping is not None:
        output = np.stack(output, 1)
        return np.apply_along_axis(lambda x: mapping[tuple(x)], axis=1, arr=output).reshape(-1, 1)
    return np.stack(output, 1)

```

Despite the short length, it is much more manageable from a production standpoint. Then our code using the EBM can be simplified and decoupled. 

>  TODO: Align with `ebm2onnx/convert.py` to create interfaces


```py
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import datasets
import numpy as np

from interpret.glassbox import ExplainableBoostingRegressor

X, y = datasets.load_diabetes(return_X_y=True)
seed = 1
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=seed
)

ebm = ExplainableBoostingRegressor(random_state=seed)
ebm.fit(X_train, y_train)

ebm_config = build_config(ebm)

output = np.hstack([
    bin_mapper(X_train, bins=x['column_mapping'], columns=x['column_index'], mapping=x['scores']) for x in ebm_config
])  # shape (None, 20)
```


```py

```

This approach is easier to work with and does not depend on the interpret library! Furthermore, it can be pickled with relative ease without package management. By using the `mapping` it will generate a 1-D learned embedding in the tabular setting without deep learning.

This approach can also be used to extend categorical variables to have a similar encoding scheme without one-hot encoding. This would transform a categorical column to a learned non-linear representation. This becomes a direct value-mapping problem instead. 

This approach can even be used to determine mappings of interaction variables without exploding the underlying dimensional space without using constructs like "feature cross". The issue with feature cross or the `PolynomialFeatures` creation from `scikit-learn` leads to an exponential increase in variables which makes it difficult to optimise without greedy search to limit the number of features in their model. 