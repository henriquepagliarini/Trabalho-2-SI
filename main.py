from dataset import DataSet
from cart import Cart
from rn import Rn
import matplotlib.pyplot as plt

def main():
    dataset = DataSet('./datasets/vict/10000v/data.csv')
    dataset.load_data()
    
    print("Treinando CART...")
    cart = Cart(dataset, 3)
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
    
    cart.retrain()
    rn.retrain()
    
    print("\nSalvando CART...")
    cart.save_model()
    print("\nSalvando RN...")
    rn.save_model()
    
    test_dataset = DataSet('./datasets/vict/1300v/data.csv')
    test_dataset.load_data()
    
    cart.test(test_dataset)
    rn.test(test_dataset)

    print(
        f"Ganho {'do CART' if cart.best_model_mse < rn.best_model_mse else 'da RN'} "
        f"em relação {'à RN' if cart.best_model_mse < rn.best_model_mse else 'ao CART'}: "
        f"{compare_mse(cart.best_model_mse, rn.best_model_mse):.2f}%"
    )
    
    print(
        f"Ganho {'do CART' if cart.best_model_rmse < rn.best_model_rmse else 'da RN'} "
        f"em relação {'à RN' if cart.best_model_rmse < rn.best_model_rmse else 'ao CART'}: "
        f"{compare_mse(cart.best_model_rmse, rn.best_model_rmse):.2f}%"
    )
        
    plt.show()
    
def compare_mse(cart_se, rn_se):
    if (cart_se == 0 and rn_se == 0):
        return -1
    
    if (cart_se < rn_se):
        return (1 - (cart_se/rn_se)) * 100
    else:
        return (1 - (rn_se/cart_se)) * 100
    
if __name__ == '__main__':
    main()