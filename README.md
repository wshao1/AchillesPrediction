# AchillesPrediction

The repository contains the work in the paper "Predicting gene knockout effects from expression data"

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
_, avg_rmse, avg_pearson, _, _, _, model = run_on_target(essentiality_file, expression_file, target_gene_name,
                                           "xg_boost", None, num_folds=1, return_model=True)
model.predict(test_expression_data)
```
