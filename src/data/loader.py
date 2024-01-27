from pandas import DataFrame
from pandas_datareader import data

class MoexData:
	def __init__(self, ticker: str):
		self.ticker = ticker

	def get(self, date_start: str, date_end: str) -> DataFrame | None:
		try:
			df: DataFrame = data.DataReader(self.ticker, 'moex', date_start, date_end)
			# df['SMA5 Open'] = df['OPEN'].rolling(5).mean()
			# df['SMA12 Open'] = df['OPEN'].rolling(12).mean()
			# df['EWMA'] = df['OPEN'].ewm(com=5).mean()
			return df
		except ImportError:
			assert False, 'bad try to pars dataframe'