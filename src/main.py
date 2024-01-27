from warnings import filterwarnings
from src.model.network import NeuronNetwork
from src.model.settings import settings

from src.data.loader import MoexData
from src.data.preparation import DataPreparation

filterwarnings('ignore')

class Main(object):
	@staticmethod
	def main() -> None:
		# Data
		moex = MoexData(settings.stock_name)
		train_data = moex.get(settings.train_date_start, settings.train_date_end)
		prediction_data = moex.get(settings.prediction_date_start, settings.prediction_date_end)

		prepared_date = DataPreparation(ticker_data_train=train_data, ticker_data_preparation=prediction_data)
		# 4 np.arrays (train_open, train_close, prediction_open, prediction_close)
		net_np_data = prepared_date.get_np_arrays()
		print(type(net_np_data[0]))
		print(net_np_data)
		# Start network learning
		NeuronNetwork(
			train_open=net_np_data[0], train_close=net_np_data[1],
			prediction_open=net_np_data[2], prediction_close=net_np_data[3]
		).prepare_model()

if __name__ == '__main__':
	Main.main()
