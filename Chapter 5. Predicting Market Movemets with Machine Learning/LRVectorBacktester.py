import numpy as np
import pandas as pd

class LRVectorBacktester(object):
    def __init__(self, symbol, start, end, amount, tc):
        self.symbol = symbol
        self.start = start
        self.end = end
        self.amount = amount
        self.tc = tc
        self.results = None
        self.get_data()

    def get_data(self):
        raw = pd.read_csv('http://hilpisch.com/pyalgo_eikon_eod_data.csv',
                          index_col = 0, parse_dates = True).dropna()
        raw = pd.DataFrame(raw[self.symbol])
        raw = raw.loc[self.start : self.end]
        raw.rename(columns = {self.symbol : 'price'}, inplace = True)
        raw['returns'] = np.log(raw / raw.shift(1))
        self.data = raw.dropna()

    def select_data(self, start, end):
        data = self.data[(self.data.index >= start) & (self.data.index <= end)].copy()
        return data

    def prepare_lags(self, start, end):
        data = self.select_data(start, end)
        self.cols = []
        for lag in range(1, self.lags + 1):
            col = f'lag_{lag}'
            data[col] = data['returns'].shift(lag)
            self.cols.append(col)
        data.dropna(inplace = True)
        self.lagged_data = data

    def fit_model(self, start, end):
        self.prepare_lags(start, end)
        reg = np.linalg.lstsq(self.lagged_data[self.cols],
                              np.sign(self.lagged_data['returns']),
                              rcond=None)[0]
        self.reg = reg

    def run_strategy(self, start_in, end_in, start_out, end_out, lags = 3):
        self.lags = lags
        self.fit_model(start_in, end_in)
        self.results = self.select_data(start_out, end_out).iloc[lags:]
        self.prepare_lags(start_out, end_out)
        prediction = np.sign(np.dot(self.lagged_data[self.cols], self.reg))
        self.results['prediction'] = prediction
        self.results['strategy'] = self.results['prediction'] * self.results['returns']
        trades = self.results['prediction'].diff().fillna(0) != 0
        self.results['strategy'][trades] -= self.tc
        self.results['creturns'] = self.amount * self.results['returns'].cumsum().apply(np.exp)
        self.results['cstrategy'] = self.amount * self.results['strategy'].cumsum().apply(np.exp)
        aperf = self.results['cstrategy'].iloc[-1]
        operf = aperf - self.results['creturns'].iloc[-1]
        return round(aperf, 2), round(operf, 2),

    def plot_result(self):
        if self.results is None:
            print("No results to plot yet. Run a strategy")
        title = '%s | TC = %.4f' % (self.symbol, self.tc)
        self.results[['creturns', 'cstrategy']].plot(title = title, figsize = [10, 6])


if __name__ == '__main__':
    lrbt = LRVectorBacktester('.SPX', '2010-1-1', '2018-06-29', 10000, 0.0)
    print(lrbt.run_strategy('2010-1-1', '2019-12-31', '2010-1-1', '2019-12-31'))
    print(lrbt.run_strategy('2010-1-1', '2015-12-31', '2016-1-1', '2019-12-31'))
    lrbt = LRVectorBacktester('GDX', '2010-1-1', '2019-12-31', 10000, 0.001)
    print(lrbt.run_strategy('2010-1-1', '2019-12-31', '2010-1-1', '2019-12-31', lags=5))
    print(lrbt.run_strategy('2010-1-1', '2015-12-31', '2016-1-1', '2019-12-31', lags=5))