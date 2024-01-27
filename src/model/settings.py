from datetime import datetime, timedelta

class Settings:
	"""
		Network settings...
	"""
	used_columns: list[str] = ['OPEN', 'CLOSE']
	stock_name: str = 'GAZP'
	date_format: str = '%Y-%m-%d'
	count_weeks_ago: int = 52
	prediction_date_end: datetime = datetime.now()
	prediction_date_start: datetime = prediction_date_end - timedelta(weeks=count_weeks_ago)

	train_date_start: datetime = prediction_date_start - timedelta(weeks=count_weeks_ago)
	train_date_end: datetime = prediction_date_start


settings = Settings()
