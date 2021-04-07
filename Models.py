from sklearn.linear_model import LassoCV
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel
from xgboost import XGBRegressor
from sklearn.utils import resample
import numpy as np
from FeatureSelection import get_features
from scipy.stats import pearsonr
import tensorflow as tf
from tensorflow.keras.layers.experimental import preprocessing
from tensorflow.python.keras.applications.densenet import layers
from data_helper import get_intersecting_gene_ids_and_data
from abc import ABC, abstractmethod
import argparse
import datetime
import sys
import warnings


def get_rmse(pred, true):
    return (sum((pred - true) ** 2) / len(true)) ** 0.5


class KNNFeatureModel(ABC):
    use_knn = None
    knn_model = None
    sclr_knn = None

    def __init__(self):
        super().__init__()

    @abstractmethod
    def train_inner(self, X, y):
        pass

    @abstractmethod
    def predict_inner(self, X, use_std):
        pass

    def predict(self, X):
        if self.use_knn:
            X = self.enrich_with_knn(X)
        return self.predict_inner(X)

    def predict_with_std(self, X):
        if self.use_knn:
            X = self.enrich_with_knn(X)
        return self.predict_inner(X, use_std=True)

    def train(self, X, y, use_knn):
        self.use_knn = use_knn
        if use_knn:
            X = self.add_knn_model(X, y)
        return self.train_inner(X, y)

    def enrich_with_knn(self, X):
        knn_out = self.knn_model.predict(X)
        # A = self.knn_model.model.kneighbors_graph(X)
        X = np.hstack((X, np.array([knn_out]).T))
        return X

    def add_knn_model(self, x_train, train_y):
        knn_model = train_model(x_train, train_y, 'knn', False)
        self.knn_model = knn_model
        x_train = self.enrich_with_knn(x_train)
        return x_train


class XgBoost:
    model = None

    def __init__(self):
        self.model = None

    def train(self, X, y):
        x_train, x_validation, y_train, y_validation = train_test_split(X, y, test_size=0.1, random_state=0)
        max_depth_l = [5]
        n_estimators_l = [500]
        learning_rate_l = [0.1, 0.2, 0.05]
        min_rmse = 10000
        min_m = -1
        min_n_est = -1
        min_lr = -1
        eval_set = [(x_train, y_train), (x_validation, y_validation)]
        for m in max_depth_l:
            for n_estimator in n_estimators_l:
                for lr in learning_rate_l:
                    model = XGBRegressor(max_depth=m, seed=0, n_estimators=n_estimator, learning_rate=lr, n_jobs=1)
                    model = model.fit(x_train, y_train, eval_set=eval_set, early_stopping_rounds=40, verbose=False)
                    val_pred = model.predict(x_validation).flatten()
                    rmse = get_rmse(val_pred, y_validation)
                    if rmse < min_rmse:
                        min_m = m
                        min_lr = lr
                        min_n_est = n_estimator
                        min_rmse = rmse

        model = XGBRegressor(max_depth=min_m, seed=0, n_estimators=min_n_est, learning_rate=min_lr, n_jobs=1)
        model = model.fit(x_train, y_train, eval_set=eval_set, early_stopping_rounds=40, verbose=False)
        self.model = model

    def predict(self, X):
        return self.model.predict(X).flatten()


class LinearModel:
    model = None
    sclr = None

    def __init__(self):
        self.model = None
        self.sclr = None

    def train(self, X, y):
        self.sclr = StandardScaler()
        self.sclr = self.sclr.fit(X)
        X = self.sclr.transform(X)
        model = LassoCV(cv=3, random_state=0)
        self.model = model.fit(X, y)

    def predict(self, X):
        X = self.sclr.transform(X)
        return self.model.predict(X).flatten()


class GaussianProcessRegressionModel(KNNFeatureModel):
    model = None
    sclr = None

    def train_inner(self, X, y):
        self.sclr = StandardScaler()
        kernel = RBF()  # + WhiteKernel()
        self.sclr = self.sclr.fit(X)
        X = self.sclr.transform(X)
        model = GaussianProcessRegressor(kernel=kernel, random_state=0)
        self.model = model.fit(X, y)

    # def train(self, X, y, use_knn=False):
    #     if use_knn:
    #         X = self.add_knn_model(X, y)
    #     self.sclr = StandardScaler()
    #     kernel = RBF()# + WhiteKernel()
    #     self.sclr = self.sclr.fit(X)
    #     X = self.sclr.transform(X)
    #     model = GaussianProcessRegressor(kernel=kernel, random_state=0)
    #     self.model = model.fit(X, y)

    def predict_inner(self, X, use_std=False):
        X = self.sclr.transform(X)
        return self.model.predict(X, return_std=use_std)

    # def enrich_with_knn(self, X):
    #     knn_out = self.knn_model.predict(X)
    #     X = np.hstack((X, np.array([knn_out]).T))
    #     return X


class DeepLearning:
    model = None

    def __init__(self):
        self.model = None

    def build_and_compile_model(self, norm, l2_reg=0.0001):
        regularizer = tf.keras.regularizers.L2(
            l2=l2_reg
        )
        model = tf.keras.Sequential([
            norm,
            layers.Dense(50, activation='relu', kernel_regularizer=regularizer),
            layers.Dropout(0.4),
            layers.Dense(20, activation='relu', kernel_regularizer=regularizer),
            layers.Dropout(0.2),
            layers.Dense(15, activation='relu', kernel_regularizer=regularizer),
            layers.Dropout(0.1),
            layers.Dense(12, activation='relu', kernel_regularizer=regularizer),
            layers.Dense(1, activation='linear')
        ])

        model.compile(loss='mean_squared_error',
                      optimizer=tf.keras.optimizers.Adam(0.001))
        return model

    def train(self, X, y):
        x_train, x_validation, y_train, y_validation = train_test_split(X, y, test_size=0.1, random_state=0)
        normalizer = preprocessing.Normalization(input_shape=[x_train.shape[1], ])
        normalizer.adapt(np.array(X))
        dnn_model = self.build_and_compile_model(normalizer)
        callback = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=100)
        validation_data = (x_validation, y_validation)
        history = dnn_model.fit(
            x_train, y_train,
            validation_split=0.1, validation_data=validation_data,
            verbose=0, epochs=1500, callbacks=[callback])
        self.model = dnn_model

    def predict(self, X):
        return self.model.predict(X).flatten()


class KNNModel:
    model = None

    def train(self, X, y, k=3):
        model = KNeighborsRegressor(n_neighbors=k, weights='distance')
        self.model = model.fit(X, y)

    def predict(self, X):
        return self.model.predict(X).flatten()


class Ensemble:
    members = []

    def __init__(self):
        self.members = []

    def train(self, X, y):
        n_splits = 4
        num_rows = X.shape[0]
        train_rows = int(num_rows * 0.9)
        is_xg_boost = True
        for _ in range(n_splits):
            # select indexes
            ix = [i for i in range(num_rows)]
            train_ix = resample(ix, replace=True, n_samples=train_rows)
            test_ix = [x for x in ix if x not in train_ix]
            train_ix = sorted(list(set(train_ix)))
            # select data
            trainX = X[train_ix, :]
            trainy = y[train_ix]
            testX = X[test_ix, :]
            testy = y[test_ix]
            # evaluate model
            if is_xg_boost:
                cur_model = XgBoost()
                cur_model.train(trainX, trainy)
                is_xg_boost = False
            else:
                cur_model = DeepLearning()
                cur_model.train(trainX, trainy)
                is_xg_boost = True
            # print('>%.3f' % test_acc)
            # scores.append(test_acc)
            self.members.append(cur_model)

    def predict(self, X):
        yhats = [model.predict(X).flatten() for model in self.members]
        yhats = np.array(yhats)
        # sum across ensemble members
        summed = np.sum(yhats, axis=0)
        # argmax across classes
        result = summed / len(self.members)
        return result


class ChooseBest:
    model = None

    def __init__(self):
        self.model = None

    def train(self, X, y):
        x_train, x_validation, y_train, y_validation = train_test_split(X, y, test_size=0.1, random_state=0)
        min_rmse = 10000
        min_model = ""
        for model_name, model_method in model_train_method_for_choose_best.items():
            m = model_method(x_train, y_train)
            vals_pred = m.predict(x_validation)
            val_rmse = get_rmse(vals_pred, y_validation)
            if val_rmse < min_rmse:
                min_rmse = val_rmse
                min_model = model_name

        self.model = model_train_method[min_model](X, y)

    def predict(self, X):
        return self.model.predict(X)


def train_linear(X, y, use_knn=False):
    m = LinearModel()
    m.train(X, y)
    return m


def train_deep_learning(X, y, use_knn=False):
    m = DeepLearning()
    m.train(X, y)
    return m


def train_xgboost(X, y, use_knn=False):
    m = XgBoost()
    m.train(X, y)
    return m


def train_gp(X, y, use_knn=True):
    m = GaussianProcessRegressionModel()
    m.train(X, y, use_knn)
    return m


def train_knn(X, y, use_knn=False):
    m = KNNModel()
    m.train(X, y)
    return m


def train_ensemble(X, y, use_knn=False):
    m = Ensemble()
    m.train(X, y)
    return m


def train_best_using_validation(X, y, use_knn=False):
    m = ChooseBest()
    m.train(X, y)
    return m


model_train_method = {
    'linear': train_linear,
    'xg_boost': train_xgboost,
    'deep': train_deep_learning,
    'ensemble': train_ensemble,
    'GP': train_gp,
    'choose_best': train_best_using_validation,
    'knn': train_knn
}
model_train_method_for_choose_best = {
    'linear': train_linear,
    'xg_boost': train_xgboost,
    'ensemble': train_ensemble,
    'GP': train_gp,
}


def train_model(X, y, model_name, use_knn=False):
    """Trains a ML model to predict y based on X input.

       Parameters
       ----------
       X : pd.DataFrame
           input data used for training

       y : np.array with shape (1, n)
           the target variable

       model_name : string
           The name of the type of model desired to train.
           Options are 'linear', 'xg_boost', 'deep', 'ensemble', 'choose_best', 'GP', 'knn'
       """
    return model_train_method[model_name](X, y, use_knn)


def cross_validation_eval(achilles_effect, expression_dat, target_gene_name, cross_validation_df, model_name,
                          achilles_id_name='DepMap_ID', expression_id_name='Unnamed: 0'):
    """Trains a ML model to predict y based on X input using cross validation
        and prints the final cross validated pearson correlations and RMSE.

           Parameters
           ----------
           achilles_effect : pd.DataFrame
               contains at least two columns, cell id column and target gene achilles scores

           expression_dat : pd.DataFrame
               expression data of all genes to be used for input to ML

           target_gene_name: String
                name of target gene column in achilles_effect dataframe

           cross_validation_df : pd.DataFrame
                columns represent cell ids except for the first column which represents which rows
                are train and which rows are test

           model_name : string
               The name of the type of model desired to train.
               Options are 'linear', 'xg_boost', 'deep', 'ensemble', 'choose_best', 'GP'

           achilles_id_name : string
               The column name of cell line id column in the achilles data set

           expression_id_name : string
               The column name of cell line id column in the expression data set
           """
    test_start_idx = 0
    for state in list(cross_validation_df.state):
        if state == "test":
            break
        test_start_idx += 1
    rmse_sum = 0
    fold_count = 0
    pearson_corr_pred_sum = 0
    model_failed = False
    for fold_col in cross_validation_df.columns[1:]:
        fold_count += 1
        cur_ids = list(cross_validation_df[fold_col])
        train_ids = set(cur_ids[0:test_start_idx])
        test_ids = set(cur_ids[test_start_idx:])
        train_achilles = achilles_effect.loc[achilles_effect[achilles_id_name].isin(train_ids)]
        test_achilles = achilles_effect.loc[achilles_effect[achilles_id_name].isin(test_ids)]
        train_achilles = train_achilles.sort_values(by=['DepMap_ID'])
        test_achilles = test_achilles.sort_values(by=['DepMap_ID'])
        train_y = train_achilles[target_gene_name]
        test_y = test_achilles[target_gene_name]
        train_expression = expression_dat.loc[expression_dat[expression_id_name].isin(train_ids)]
        test_expression = expression_dat.loc[expression_dat[expression_id_name].isin(test_ids)]
        train_expression = train_expression.sort_values(by=['Unnamed: 0'])
        test_expression = test_expression.sort_values(by=['Unnamed: 0'])
        expression_feature_indices = get_features(train_y, train_expression, 20)
        in_use_gene_names = train_expression.columns[expression_feature_indices]
        x_train = train_expression[in_use_gene_names]
        x_train = np.array(x_train)
        train_y = np.array(train_y)
        x_test = test_expression[in_use_gene_names]
        x_test = np.array(x_test)
        test_y = np.array(test_y)
        try:
            model = train_model(x_train, train_y, model_name)
            test_pred = model.predict(x_test)
            rmse = get_rmse(test_pred, test_y)
            pred_corr, _ = pearsonr(test_pred, test_y)
            rmse_sum += rmse
            pearson_corr_pred_sum += pred_corr
            print("{}: {} with pearson corr {}".format(str(datetime.datetime.now()), fold_col, pred_corr))
        except Exception as inst:
            print("Exception on {} with fold {}".format(target_gene_name, fold_col))
            print(str(inst))
            model_failed = True
    if not model_failed:
        return rmse_sum / fold_count, pearson_corr_pred_sum / fold_count
    else:
        return -1, -1


def handle_nans(x_train, y_train):
    indices_where_nan = np.argwhere(np.isnan(y_train)).flatten()
    y_train = np.delete(y_train, indices_where_nan)
    x_train = np.delete(x_train, indices_where_nan, axis=0)
    return x_train, y_train


def train_no_eval(achilles_effect, expression_dat, target_gene_name, model_name,
                     achilles_id_name='DepMap_ID', expression_id_name='Unnamed: 0'):
    """Trains a ML model to predict y based on X input using a train/test split

           Parameters
           ----------
           achilles_effect : pd.DataFrame
               contains at least two columns, cell id column and target gene achilles scores

           expression_dat : pd.DataFrame
               expression data of all genes to be used for input to ML

           target_gene_name: String
                name of target gene column in achilles_effect dataframe

           model_name : string
               The name of the type of model desired to train.
               Options are 'linear', 'xg_boost', 'deep', 'ensemble', 'choose_best', 'GP'

           achilles_id_name : string
               The column name of cell line id column in the achilles data set

           expression_id_name : string
               The column name of cell line id column in the expression data set
           """
    achilles_effect = achilles_effect.sort_values(by=['DepMap_ID'])
    y = achilles_effect[target_gene_name]
    expression_dat = expression_dat.sort_values(by=['Unnamed: 0'])
    expression_feature_indices = get_features(y, expression_dat, 10)
    in_use_gene_names = expression_dat.columns[expression_feature_indices]
    in_use_gene_names = sorted(in_use_gene_names)
    x_train = expression_dat[in_use_gene_names]
    x_train = np.array(x_train)
    train_y = np.array(y)
    x_train, train_y = handle_nans(x_train, train_y)
    model = train_model(x_train, train_y, model_name, True)
    return model, in_use_gene_names


def train_test_eval(achilles_effect, expression_dat, target_gene_name, train_test_df, model_name,
                     achilles_id_name='DepMap_ID', expression_id_name='Unnamed: 0', use_knn=False):
    """Trains a ML model to predict y based on X input using a train/test split
        and prints the final cross validated pearson correlations and RMSE.

           Parameters
           ----------
           achilles_effect : pd.DataFrame
               contains at least two columns, cell id column and target gene achilles scores

           expression_dat : pd.DataFrame
               expression data of all genes to be used for input to ML

           target_gene_name: String
                name of target gene column in achilles_effect dataframe

           train_test_df : pd.DataFrame
                columns represent cell ids except for the first column which represents which rows
                are train and which rows are test

           model_name : string
               The name of the type of model desired to train.
               Options are 'linear', 'xg_boost', 'deep', 'ensemble', 'choose_best', 'GP'

           achilles_id_name : string
               The column name of cell line id column in the achilles data set

           expression_id_name : string
               The column name of cell line id column in the expression data set
           """
    test_start_idx = 0
    for state in list(train_test_df.train_test_split):
        if state == "test":
            break
        test_start_idx += 1
    rmse_sum = 0
    fold_count = 0
    pred_p_val = -1
    pearson_corr_pred_sum = 0
    model_failed = False
    fold_count += 1
    cur_ids = list(train_test_df.id)
    train_ids = set(cur_ids[0:test_start_idx])
    test_ids = set(cur_ids[test_start_idx:])
    train_achilles = achilles_effect.loc[achilles_effect[achilles_id_name].isin(train_ids)]
    test_achilles = achilles_effect.loc[achilles_effect[achilles_id_name].isin(test_ids)]
    train_achilles = train_achilles.sort_values(by=['DepMap_ID'])
    test_achilles = test_achilles.sort_values(by=['DepMap_ID'])
    train_y = train_achilles[target_gene_name]
    test_y = test_achilles[target_gene_name]
    train_expression = expression_dat.loc[expression_dat[expression_id_name].isin(train_ids)]
    test_expression = expression_dat.loc[expression_dat[expression_id_name].isin(test_ids)]
    train_expression = train_expression.sort_values(by=['Unnamed: 0'])
    test_expression = test_expression.sort_values(by=['Unnamed: 0'])
    expression_feature_indices = get_features(train_y, train_expression, 20)
    in_use_gene_names = train_expression.columns[expression_feature_indices]
    x_train = train_expression[in_use_gene_names]
    x_train = np.array(x_train)
    train_y = np.array(train_y)
    x_test = test_expression[in_use_gene_names]
    x_test = np.array(x_test)
    test_y = np.array(test_y)
    x_train, train_y = handle_nans(x_train, train_y)
    x_test, test_y = handle_nans(x_test, test_y)
    try:
        model = train_model(x_train, train_y, model_name, use_knn=use_knn)
        test_pred = model.predict(x_test)
        rmse = get_rmse(test_pred, test_y)
        pred_corr, pred_p_val = pearsonr(test_pred, test_y)
        rmse_sum += rmse
        pearson_corr_pred_sum += pred_corr
        print("{}: {} with test pearson corr {}".format(str(datetime.datetime.now()), "train/test split", pred_corr))
    except Exception as inst:
        print("Exception on {} with {}".format(target_gene_name, "train/test split"))
        print(str(inst))
        model_failed = True
    if not model_failed:
        return rmse_sum / fold_count, pearson_corr_pred_sum / fold_count, pred_p_val
    else:
        return -1, -1, -1


def run_on_target(gene_effect_file_name, gene_expression_file_name, target_gene_name, model_name, log_output,
                  num_folds=5,
                  cv_df_file_name=None, train_test_df_file_name=None, return_model=False, genes_for_features=None,
                  use_knn=False):
    to_print = "{}: Beginning processing gene {}".format(str(datetime.datetime.now()), target_gene_name)
    if log_output is not None:
        print(to_print, file=open(log_output, 'w'))
    else:
        print(to_print)
    if not sys.warnoptions:
        warnings.simplefilter("ignore")
    achilles_scores, gene_expression, \
    train_test_df, cv_df = get_intersecting_gene_ids_and_data(gene_effect_file_name,
                                                              gene_expression_file_name,
                                                              cv_df_file=cv_df_file_name,
                                                              train_test_df_file=train_test_df_file_name,
                                                              num_folds=num_folds)
    achilles_id_col_name = 'DepMap_ID'
    expression_id_col_name = 'Unnamed: 0'
    achilles_scores = achilles_scores[[achilles_id_col_name, target_gene_name]]
    if genes_for_features:
        intersection_genes = list(genes_for_features.intersection(set(gene_expression.columns))) + [expression_id_col_name]
        gene_expression = gene_expression[intersection_genes]
    model = None
    features = None
    if num_folds == 0:
        model, features = train_no_eval(achilles_scores, gene_expression, target_gene_name,
                                                    model_name)
        cv_rmse = None
        cv_pearson = None
        pearson_p_val = None
    elif num_folds > 1:
        cv_rmse, cv_pearson = cross_validation_eval(achilles_scores, gene_expression, target_gene_name, cv_df,
                                                    model_name)
        pearson_p_val = None
    else:
        cv_rmse, cv_pearson, pearson_p_val = train_test_eval(achilles_scores, gene_expression, target_gene_name, train_test_df,
                                                    model_name, use_knn=use_knn)
    if return_model:
        return target_gene_name, cv_rmse, cv_pearson, pearson_p_val, model, features
    else:
        return target_gene_name, cv_rmse, cv_pearson, pearson_p_val, None, None


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gene_effect',
                        default='Achilles_gene_effect.csv')
    parser.add_argument('--gene_expression',
                        default='CCLE_expression.csv')
    parser.add_argument('--target_gene_name',
                        default='CHMP1A')
    parser.add_argument('--model_name', help="Options are 'linear', 'xg_boost', 'deep', 'ensemble', 'choose_best', "
                                             "'GP'",
                        default='GP')
    parser.add_argument('--num_folds', help="Cross validation folds. Default is train/test, i.e. 1",
                        default=1)
    parser.add_argument('--cv_file', help="Cross validation ids file path. See data_helper.py for how to create such "
                                          "a file.",
                        default="cross_validation_folds_ids.tsv")
    parser.add_argument('--train_test_file', help="train/test ids file path. See data_helper.py for how to create"
                                                  " such a file.",
                        default="train_test_split.tsv")
    parser.add_argument('--log_output', help="A filename. default output is to std.out",
                        default=None)
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    _, cv_rmse, cv_pearson, _, _, _ = run_on_target(args.gene_effect, args.gene_expression, args.target_gene_name,
                                           args.model_name, args.log_output, args.num_folds, args.cv_file,
                                                    args.train_test_file, use_knn=True)
    print("rmse " + str(cv_rmse))
    print("cv_pearson " + str(cv_pearson))
