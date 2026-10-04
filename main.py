from dataset import DataSet

def main():
    dataset = DataSet('./datasets/vict/10000v/data.csv')
    dataset.load_data()
    
if __name__ == '__main__':
    main()