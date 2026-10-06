from dataset import DataSet
from cart import Cart
from rn import Rn
import matplotlib as plt
def main():
    dataset = DataSet('./datasets/vict/10000v/data.csv')
    dataset.load_data()
    
    print("Treinando CART...")
    cart = Cart(dataset, 5)
    cart.train()
    cart.process_results()
    cart.show_results()
    cart.show_best_model()
    
    print("\nTreinando RN...")
    rn = Rn(dataset, 5)
    rn.train()
    rn.process_results()
    rn.show_results()
    rn.show_best_model()
    
if __name__ == '__main__':
    main()