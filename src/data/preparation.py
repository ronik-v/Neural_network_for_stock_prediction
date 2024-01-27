from typing import Any

from pandas import DataFrame
from numpy import array, ndarray, dtype, isnan


class DataPreparation:
	def __init__(self, ticker_data_train: DataFrame, ticker_data_preparation: DataFrame):
		self.ticker_data_train = ticker_data_train
		self.ticker_data_preparation = ticker_data_preparation

	# Get 4 np.arrays...
	def get_np_arrays(self) -> tuple[
		ndarray[Any, dtype[Any]], ndarray[Any, dtype[Any]], ndarray[Any, dtype[Any]], ndarray[Any, dtype[Any]]]:
		""" Data for network | open - is open prices, close - is close prices """
		train_open, train_close = [], []
		prediction_open, prediction_close = [], []

		for day_index in range(len(self.ticker_data_train)):
			if not isnan(self.ticker_data_train['OPEN'][day_index]) and not isnan(self.ticker_data_train['CLOSE'][day_index]):
				train_open.append(self.ticker_data_train['OPEN'][day_index])
				train_close.append(self.ticker_data_train['CLOSE'][day_index])

			if not isnan(self.ticker_data_preparation['OPEN'][day_index]) and not isnan(self.ticker_data_preparation['CLOSE'][day_index]):
				prediction_open.append(self.ticker_data_preparation['OPEN'][day_index])
				prediction_close.append(self.ticker_data_preparation['CLOSE'][day_index])
		print(train_open, train_close)
		print(prediction_open, prediction_close)
		return array(train_open), array(train_close), array(prediction_open), array(prediction_close)

