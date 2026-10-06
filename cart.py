from regressor import Regressor
from sklearn.tree import DecisionTreeRegressor

class Cart(Regressor):
    def __init__(self, dataset, n_folds):
        parameters = [
            {
                # Subajustada (U)
                'max_depth': [1],
                'min_samples_leaf': [64],
                'min_samples_split': [32]
            },
            {
                # Equilibrada (E)
                'max_depth': [8],
                'min_samples_leaf': [32],
                'min_samples_split': [16]
            },
            {
                # Sobreajustada (O)
                'max_depth': [24],
                'min_samples_leaf': [1],
                'min_samples_split': [2]
            }
        ]

        model = DecisionTreeRegressor(random_state=42)
        
        super().__init__(
            dataset,
            model,
            parameters,
            n_folds
        )