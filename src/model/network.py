from typing import Any
from keras import Sequential
from keras.src.callbacks import ReduceLROnPlateau
from keras.src.layers import Dense, Dropout, BatchNormalization, LeakyReLU, Activation
from numpy import ndarray, dtype, arange, std, append

import matplotlib.pyplot as plt

from src.util import sma_prediction


class NeuronNetwork:
	def __init__(self, train_open: ndarray[Any, dtype[Any]], train_close: ndarray[Any, dtype[Any]],
				 prediction_open: ndarray[Any, dtype[Any]], prediction_close: ndarray[Any, dtype[Any]]):
		self.train_open = train_open
		self.train_close = train_close
		self.prediction_open = prediction_open
		self.prediction_close = prediction_close

		# Network settings
		self.epoch: int = 300
		self.days_indent: int = 30
		self.model: Sequential = Sequential()

	def prepare_model(self) -> None:
		self.model.add(Dense(256, input_dim=1, activation='relu'))
		self.model.add(Dropout(0.2))
		self.model.add(Dense(128, input_dim=1, activation='relu'))
		self.model.add(Dropout(0.2))
		self.model.add(Dense(64, input_dim=1, activation='relu'))
		self.model.add(Dropout(0.2))
		self.model.add(Dense(32, input_dim=1, activation='relu'))
		self.model.add(Dropout(0.2))
		self.model.add(Dense(16, input_dim=1, activation='relu'))
		self.model.add(Dropout(0.2))
		self.model.add(BatchNormalization())
		self.model.add(LeakyReLU())
		self.model.add(Dense(1))
		self.model.add(Activation('relu'))
		reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.65, patience=5, min_lr=0.0001)
		self.model.compile(optimizer='rmsprop', loss='mean_squared_error', metrics=['mean_absolute_error'])

		history = self.model.fit(self.train_open, self.train_close,
								epochs=self.epoch,
								batch_size=4,
								verbose=1,
								validation_data=(self.prediction_open[:len(self.prediction_open) - self.days_indent], self.prediction_close[:len(self.prediction_close) - self.days_indent]),
								shuffle=True,
								callbacks=[reduce_lr], use_multiprocessing=True)

		last_predictions = self.model.predict(self.prediction_open[:len(self.prediction_open) - self.days_indent])
		last_predictions_close: list[float] = [last_predictions[i][0] for i in range(len(last_predictions))]
		fact_diff_close: list[float] = [abs(f_close - l_close) for f_close, l_close in zip(self.prediction_open[:len(self.prediction_open) - self.days_indent], last_predictions_close)]
		std_prediction = float(std(fact_diff_close))

		# append sma
		sma_add = sma_prediction(self.prediction_open.tolist(), 12, 3)
		self.prediction_open = append(self.prediction_open, sma_add)

		predictions = self.model.predict(self.prediction_open[len(self.prediction_open) - self.days_indent:])
		predictions_close: list[float] = [predictions[i][0] + std_prediction for i in range(len(predictions))]
		day_index_prediction: list[int] = [day_index for day_index in range(len(self.prediction_open) - self.days_indent, len(self.prediction_open))]
		print(f'predictions = \n{predictions}')
		# Plot network learning stat
		self.__plot_learning_result(history)
		# Plot network forecast close prices
		self.__plot_forecast_result(predictions_close, self.prediction_open[len(self.prediction_open) - self.days_indent:], day_index_prediction)

	# Plot
	def __plot_learning_result(self, _history) -> None:
		arr_size = arange(0, self.epoch)
		plt.style.use('ggplot')
		plt.figure()
		plt.plot(arr_size, _history.history['loss'], label='loss')
		plt.plot(arr_size, _history.history['val_loss'], label='val_loss')
		plt.plot(arr_size, _history.history['mean_absolute_error'], label='mean_absolute_error')
		plt.plot(arr_size, _history.history['val_mean_absolute_error'], label='val_mean_absolute_error')
		plt.title('Model errors for ticker prediction')
		plt.xlabel('epoch')
		plt.ylabel('Loss/mean_absolute_error')
		plt.legend()
		plt.grid(True)
		plt.show()

	def __plot_forecast_result(self, prediction_data, fact_data, day_index_prediction) -> None:
		plt.style.use('ggplot')
		plt.plot(day_index_prediction, prediction_data, label='Predictions close prices')
		plt.plot(day_index_prediction, fact_data, label='Fact close prices')
		plt.title('Predictions')
		plt.xlabel('Day index')
		plt.ylabel('Close price')
		plt.legend()
		plt.grid(True)
		plt.show()