import pandas as pd
#import dataset
train = pd.read_csv('train_set.csv',sep=',',header=0)
test = pd.read_csv('test_set.csv',sep=',',header=0)
#cleaning the train set
train.dropna(subset=['target'],inplace=True)
print(train.isna().sum().sum())
print(test.isna().sum().sum())
X_raw = train.drop('target', axis=1)
y_raw =train[train.columns[-1]]
# check if there are only floats and ints
X_raw.dtypes.nunique()
#drop features with a single value
X_raw_nuniq = pd.DataFrame([X_raw.nunique()])
feature_nuniq = X_raw_nuniq[X_raw_nuniq == 1].dropna(axis=1).columns
X_raw = X_raw.drop(columns= feature_nuniq.to_list())
test_raw_nuniq = pd.DataFrame([test.nunique()])
test_feature_nuniq = test_raw_nuniq[test_raw_nuniq == 1].dropna(axis=1).columns
test_raw = test.drop(columns= test_feature_nuniq.to_list())

del X_raw_nuniq, feature_nuniq, test_raw_nuniq,test_feature_nuniq

# pca
from sklearn.decomposition import KernelPCA, IncrementalPCA,PCA, TruncatedSVD
import numpy as np
pca = PCA(n_components=20)

# decreasing dataframe size enough to not throw MemoryError
X_pca = pca.fit_transform(X_raw,y_raw)
#vedo quanto i primi n_component PC spiegano
print(pca.explained_variance_ratio_.sum())


# n componenti
n_pcs= pca.components_.shape[0]
# get the index of the most important feature on EACH component
most_important = [np.abs(pca.components_[i]).argmax() for i in range(n_pcs)]
most_important_names = [X_raw.columns.to_list()[most_important[i]] for i in range(n_pcs)]
# LIST COMPREHENSION HERE AGAIN
dic = {'PC{}'.format(i): most_important_names[i] for i in range(n_pcs)}
# build the dataframe
df = pd.DataFrame(dic.items())

X = X_raw[list(set(most_important_names))]
X_test = test[list(set(most_important_names))]
#smote
from imblearn.over_sampling import SMOTE
smote = SMOTE(sampling_strategy='minority',random_state=42)

X_smote,y_smote = smote.fit_resample(np.array(X),y_raw.ravel())
X_smote = pd.DataFrame(X_smote,columns=X.columns)
y_smote = pd.DataFrame(y_smote)
from sklearn.model_selection import train_test_split
train_x, valid_x, train_y, valid_y = train_test_split(X_smote, y_smote, test_size=0.25, random_state=10,
                                                      shuffle=True)

import hyperopt
import sys
import catboost.utils as cbu
import catboost as cb
class catboost_classifier(object):
    def __init__(self, dataset, const_params, fold_count):
        self._dataset = dataset
        self._const_params = const_params.copy()
        self._fold_count = fold_count
        self._evaluated_count = 0

    def _to_catboost_params(self, hyper_params):
        return {
            'learning_rate': hyper_params['learning_rate'],

            'l2_leaf_reg': hyper_params['l2_leaf_reg'],
               'max_depth':hyper_params['max_depth'],
               'colsample_bylevel':hyper_params['colsample_bylevel'],
               'n_estimators':hyper_params['n_estimators'],

               'boosting_type':hyper_params['boosting_type']

        }

    # hyperopt optimizes an objective using `__call__` method (e.g. by doing
    # `foo(hyper_params)`), so we provide one
    def __call__(self, hyper_params):
        # join hyper-parameters provided by hyperopt with hyper-parameters
        # provided by the user
        params = self._to_catboost_params(hyper_params)
        params.update(self._const_params)

        print('evaluating params={}'.format(params), file=sys.stdout)
        sys.stdout.flush()

        # we use cross-validation for objective evaluation, to avoid overfitting
        scores = cb.cv(
            pool=self._dataset,
            params=params,
            fold_count=self._fold_count,
            partition_random_seed=20181224,
            verbose=False)

        # scores returns a dictionary with mean and std (per-fold) of metric
        # value for each cv iteration, we choose minimal value of objective
        # mean (though it will be better to choose minimal value among all folds)
        # because noise is additive
        max_mean_auc = np.max(scores['test-AUC-mean'])
        print('evaluated score={}'.format(max_mean_auc), file=sys.stdout)

        self._evaluated_count += 1
        print('evaluated {} times'.format(self._evaluated_count), file=sys.stdout)

        # negate because hyperopt minimizes the objective
        return {'loss': -max_mean_auc, 'status': hyperopt.STATUS_OK}


def find_best_hyper_params(dataset, const_params, max_evals=100):
    # we are going to optimize these three parameters, though there are a lot more of them (see CatBoost docs)
    parameter_space = {
        'learning_rate': hyperopt.hp.choice('learning_rate', np.arange(0.01, 0.06, 0.01)),

    'max_depth': hyperopt.hp.choice('max_depth', np.arange(5, 16, 1)),
    'colsample_bylevel': hyperopt.hp.choice('colsample_bylevel', np.arange(0.3, 0.8, 0.1)),
    'n_estimators': hyperopt.hp.uniform('n_estimators', 50, 100),
    'boosting_type': hyperopt.hp.choice('boosting_type', ['Ordered', 'Plain']),
    'l2_leaf_reg': hyperopt.hp.uniform('l2_leaf_reg', 3, 8),
}
    objective = catboost_classifier(dataset=dataset, const_params=const_params, fold_count=3)
    trials = hyperopt.Trials()
    best = hyperopt.fmin(
        fn=objective,
        space=parameter_space,
        algo=hyperopt.rand.suggest,
        max_evals=max_evals,
        rstate=np.random.RandomState(seed=20181224),
    trials=trials)
    return best

best = find_best_hyper_params(train,const_params=const_params)


def train_best_model(X, y, const_params, max_evals=100, use_default=False):
    # convert pandas.DataFrame to catboost.Pool to avoid converting it on each
    # iteration of hyper-parameters optimization
    dataset = cb.Pool(X, y) #, cat_features=np.where(X.dtypes != np.float)[0])

    if use_default:
        # pretrained optimal parameters
        best = {
            'learning_rate': 0.4234185321620083,
            'depth': 5,
            'l2_leaf_reg': 9.464266235679002}
    else:
        best = find_best_hyper_params(dataset, const_params, max_evals=max_evals)

    # merge subset of hyper-parameters provided by hyperopt with hyper-parameters
    # provided by the user
    hyper_params = best.copy()
    hyper_params.update(const_params)

    # drop `use_best_model` because we are going to use entire dataset for
    # training of the final model
    hyper_params.pop('use_best_model', None)

    model = cb.CatBoostClassifier(**hyper_params)
    model.fit(dataset, verbose=False)

    return model, hyper_params

# make it True if your want to use GPU for training
have_gpu = False
# skip hyper-parameter optimization and just use provided optimal parameters
use_optimal_pretrained_params = True
# number of iterations of hyper-parameter search
hyperopt_iterations = 30

const_params = dict({
    'task_type': 'GPU' if have_gpu else 'CPU',
    'loss_function': 'Logloss',
    'eval_metric': 'Logloss',
    'iterations': 100,
    'random_seed': 20181224})

model, params = train_best_model(
    X_smote, y_smote,
    const_params,
    max_evals=hyperopt_iterations,
    use_default=use_optimal_pretrained_params)
print('best params are {}'.format(params), file=sys.stdout)






















