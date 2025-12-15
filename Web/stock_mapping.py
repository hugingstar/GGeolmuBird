import pandas as pd

class mapped():
    def __init__(self, path):
        #Stock dataframe
        df_stock = pd.read_csv(path, encoding="utf-8-sig")
        # Mapping
        df_map = df_stock.set_index('Name')['Code']
        #Dict    
        self.stock_map = self.stock_map = df_map.to_dict()

    def output(self):
        return self.stock_map

if __name__ == '__main__':
    mmp = mapped(path="C:/Users/User/PycharmProjects/TRADE/Web/stock_list.csv")

    stock_map = mmp.output()

    print(stock_map)