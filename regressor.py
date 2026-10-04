import joblib

class Regressor:
    def __init__(self, dataset, model, parameters, n_folds):
        self.dataset = dataset
        self.model = model
        self.parameters = parameters
        self.n_folds = n_folds

        self.results = {}
        self.best_model = None
        
    def train(self):
        pass
    
    def retrain(self):
        print(f"\n========= Retreino - {self.__class__.__name__.upper()} =========")
        
    def test(self, test_dataset):
        print(f"\n========= Teste - {self.__class__.__name__.upper()} =========")
    
    def save_model(self):
        filename = f"melhor_{self.__class__.__name__.lower()}.joblib"
        joblib.dump(self.best_model, filename)
        print(f"Modelo salvo em: {filename}")
    
    def process_results(self):
        pass
    
    def show_results(self):
        print(f"\n=============== Classificador {self.__class__.__name__.upper()} ===============")
        
    def show_best_model(self):
        print(f"\n=============== Melhor modelo - {self.__class__.__name__.upper()} ===============")