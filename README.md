# AchillesPrediction

The repository contains the work in the paper "Predicting gene knockout effects from expression data"

The following are the packages used in this repository:
```
dtreeviz==1.3.6
gget==0.2.1
graphviz==0.19.1
joblib==1.2.0
matplotlib==3.0.2
numpy==1.16.2
pandas==0.23.3+dfsg
scikit_learn==1.1.3
scipy==1.1.0
statsmodels==0.12.2
tensorflow==2.4.1
xgboost==1.3.3
```

The main point of entry is via Models.py main method

In order to run the code download expression data and essentiality data [here](https://depmap.org/portal/download/).
For the sake of this example we will refer to the filename/location of the expression data as `expression_file` and the essentiality data filename/location as `essentiality_file`.

In order to train a model using a train/test split and choosing the best model automatically use the following command:
```
python3 Models.py --gene_effect essentiality_file --gene_expression expression_file --target_gene_name my_target_gene  --model_name choose_best --num_folds 1
```

In order to train a model using a train/test split using XGBoost and then make a prediction within python code run the following:
```
# this is python code
from Models import run_on_target
import pandas as pd
from data_helper import clean_gene_names

_, avg_rmse, avg_pearson, _, _, features, model = run_on_target(essentiality_file, expression_file, target_gene_name,
                                                         "xg_boost", None, num_folds=1, return_model=True)

for_predict_expression = pd.read_csv(prediction_data_file)
for_predict_expression = for_predict_expression.sample(10)
for_predict_expression = clean_gene_names(for_predict_expression, for_predict_expression.columns[0])
for_predict_expression = for_predict_expression[features]
predictions = model.predict(for_predict_expression.values)
[print(x) for x in predictions]
```
