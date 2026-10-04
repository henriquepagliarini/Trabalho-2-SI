from regressor import Regressor
from sklearn.neural_network import MLPRegressor

class Rn(Regressor):
    def __init__(self, dataset, n_folds):
        # Hiperparâmetros não estão ajustados
        parameters = [
            {
                # Subajustada (U)
                'hidden_layer_sizes': [(16, 8)],
                'activation': ['relu'],
                'learning_rate_init': [0.01],
                'solver': ['adam']
            },
            {
                # Equilibrada (E)
                'hidden_layer_sizes': [(16, 8)],
                'activation': ['relu'],
                'learning_rate_init': [0.01],
                'solver': ['adam']
            },
            {
                # Sobreajustada (O)
                'hidden_layer_sizes': [(16, 8)],
                'activation': ['relu'],
                'learning_rate_init': [0.01],
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