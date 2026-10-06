from regressor import Regressor
from sklearn.tree import DecisionTreeRegressor

class Cart(Regressor):
    def __init__(self, dataset, n_folds):
        # Hiperparâmetros não estão ajustados
        parameters = [
            {
                # Subajustada (U)
                'criterion': ['squared_error'],
                'max_depth': [8],
                'min_samples_leaf': [10]
            },
            {
                # Equilibrada (E)
                'criterion': ['squared_error'],
                'max_depth': [8],
                'min_samples_leaf': [10],
            },
            {
                # Sobreajustada (O)
                'criterion': ['squared_error'],
                'max_depth': [8],
                'min_samples_leaf': [10]
            }
        ]

        model = DecisionTreeRegressor(random_state=42)
        
        super().__init__(
            dataset,
            model,
            parameters,
            n_folds
        )