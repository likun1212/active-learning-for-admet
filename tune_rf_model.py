# -*- encoding: utf-8 -*-
"""
@File    :   tune_rf_model.py
@Time    :   2022/07/01 16:03:18
@Author  :   likun.yang 
@Contact :   likun_yang@foxmail.com
@Description: 
"""

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV

# Number of trees in random forest
n_estimators = np.linspace(100, 3000, int((3000 - 100) / 200) + 1, dtype=int)
# Number of features to consider at every split
max_features = ["log2", "sqrt"]
# Maximum number of levels in tree
max_depth = [1, 5, 10, 20, 50, 75, 100, 150, 200]
# Minimum number of samples required to split a node
# min_samples_split = [int(x) for x in np.linspace(start = 2, stop = 10, num = 9)]
min_samples_split = [1, 2, 5, 10, 15, 20, 30]
# Minimum number of samples required at each leaf node
min_samples_leaf = [1, 2, 3, 4]
# Method of selecting samples for training each tree
bootstrap = [True, False]
# Criterion
criterion = ["gini", "entropy"]
random_grid = {
    "n_estimators": n_estimators,
    "max_features": max_features,
    "max_depth": max_depth,
    "min_samples_split": min_samples_split,
    "min_samples_leaf": min_samples_leaf,
    "bootstrap": bootstrap,
    "criterion": criterion,
}
rf_base = RandomForestClassifier()
rf_random = RandomizedSearchCV(
    estimator=rf_base,
    param_distributions=random_grid,
    n_iter=30,
    cv=5,
    verbose=2,
    random_state=42,
    n_jobs=4,
)
# rf_random.fit(training, training_labels)
