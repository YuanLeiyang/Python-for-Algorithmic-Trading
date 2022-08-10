['returns']
import numpy as np
import pandas as pd
from sklearn import linear_model

class ScikitVectorBacktester():
    def __init__(self, symbol, start, end, amount, tc, model):
        self.symbol = symbol
        self.start = start
        self.end = end
        self.amount = amount
        self.tc = tc
        self.results = None
        if model == 'regression':
            self.model = linear_model.LinearRegression()
        elif model == 'logistic':
            self.model = linear_model.LogisticRegression(C = 1e6, solver='lbfgs', multi_class = 'ovr', max_iter = 1000)
        else:
            raise ValueError('Model not known or not yet implemented')
        self.get_data()

    def get_data(self):
        raw = pd.read_csv('http://hilpisch.com/pyalgo_eikon_eod_data.csv',
                          index_col = 0, parse_dates=True).dropna()
        raw = pd.DataFrame(raw[self.symbol])
        raw = raw.loc[self.start: self.end]
        raw.rename(columns={self.symbol:'price'}, inplace = True)
        raw['returns'] = np.log(raw/raw.shift(1))
        self.data = raw.dropna()

    def select_data(self, start, end):
        data = self.data[(self.data.index >= start) & (self.data.index <= end)].copy()
        return data

    def prepare_features(self, start, end):
        self.data_subset = self.select_data(start, end)
        self.feature_columns = []
        for lag in range(1, self.lags + 1) :
            col = 'lag_{}'.format(lag)
            self.data_subset[col] = self.data_subset['returns'].shift(lag)
            self.feature_columns.append(col)
        self.data_subset.dropna(inplace = True)

    def fit_model(self, start, end):
        self.prepare_features(start, end)
        self.model.fit(self.data_subset[self.feature_columns],
                       np.sign(self.data_subset['returns']))

    def run_strategy(self, start_in, end_in, start_out, end_out, lags = 3):
        self.lags = lags
        self.fit_model(start_in, end_in)
        self.prepare_features(start_out, end_out)
        prediction = self.model.predict(self.data_subset[self.feature_columns])
        self.data_subset['prediction'] = prediction
        self.data_subset['strategy'] = (self.data_subset['prediction'] * self.data_subset['returns'])
        trades = self.data_subset['prediction'].diff().fillna(0) != 0

        self.data_subset['strategy'][trades] -= self.tc
        self.data_subset['creturns'] = (self.amount * self.data_subset['returns'].cumsum().apply(np.exp))
        self.data_subset['cstrategy'] = (self.amount * self.data_subset['strategy'].cumsum().apply(np.exp))

        self.results = self.data_subset

        aperf = self.results['cstrategy'].iloc[-1]

        operf = aperf - self.results['creturns'].iloc[-1]
        return round(aperf, 2), round(operf, 2),

    def plot_results(self):
        if self.results is None:
            print("No results to plot yet. Run a strategy")

        title = "%s | TC = %.4f" % (self.symbol, self.tc)

        self.results[['creturns', 'cstrategy']].plot(title = title,figsize = [10,6])

if __name__ == '__main__':
    scibt = ScikitVectorBacktester('.SPX', '2010-1-1', '2019-12-31',
                                   10000, 0.0, 'regression')
    print(scibt.run_strategy('2010-1-1', '2019-12-31',
                             '2010-1-1', '2019-12-31'))
    print(scibt.run_strategy('2010-1-1', '2016-12-31',
                             '2017-1-1', '2019-12-31'))
    scibt = ScikitVectorBacktester('.SPX', '2010-1-1', '2019-12-31',
                                   10000, 0.0, 'logistic')

    print(scibt.run_strategy('2010-1-1', '2019-12-31',
                             '2010-1-1', '2019-12-31'))
    print(scibt.run_strategy('2010-1-1', '2016-12-31',
                             '2017-1-1', '2019-12-31'))

    scibt = ScikitVectorBacktester('.SPX', '2010-1-1', '2019-12-31',
                                   10000, 0.001, 'logistic')

    print(scibt.run_strategy('2010-1-1', '2019-12-31',
                             '2010-1-1', '2019-12-31', lags=15))
    print(scibt.run_strategy('2010-1-1', '2013-12-31',
                             '2014-1-1', '2019-12-31', lags=15))



