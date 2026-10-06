from regressor import Regressor
from sklearn.neural_network import MLPRegressor

class Rn(Regressor):
    def __init__(self, dataset, n_folds):
        parameters = [
            {
                # Subajustada (U)
                'hidden_layer_sizes': [(1)],
                'activation': ['relu'],
                'learning_rate_init': [0.01],
                'solver': ['adam']
            },
            {
                # Equilibrada (E)
                'hidden_layer_sizes': [(100, 50, 25)],
                'activation': ['logistic'],
                'learning_rate_init': [0.001],
                'solver': ['adam']
            },
            {
                # Sobreajustada (O)
                'hidden_layer_sizes': [(100, 100, 50, 50)],
                'activation': ['relu'],
                'learning_rate_init': [0.0001],
                'solver': ['adam']
            }
        ]
        
        model = MLPRegressor(random_state=42, max_iter=500)

        super().__init__(
            dataset,
            model,
            parameters,
            n_folds
        )