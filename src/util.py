mean = lambda data, step: sum(data[len(data) - step:]) / step


def sma_prediction(data: list[float], step: int, count: int):
	result: list[float] = data
	for _ in range(count):
		sma_val = mean(result, step)
		result.append(sma_val)
	return result[len(result) - count:]
