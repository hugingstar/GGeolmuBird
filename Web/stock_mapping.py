import pandas as pd

class mapped():
    def __init__(self, path):
        df_stock = pd.read_csv(path, encoding="cp949")
        
        df_map = df_stock.set_index('Name')['Code']

        self.stock_map = self.stock_map = df_map.to_dict()

    def output(self):
        return self.stock_map

if __name__ == '__main__':
    mmp = mapped(path="C:/Users/User/PycharmProjects/TRADE/Web/stock_list.csv")

    stock_map = mmp.output()

    print(stock_map)