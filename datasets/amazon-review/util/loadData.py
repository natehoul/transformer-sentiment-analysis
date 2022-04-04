import pandas as pd
import gzip

import os

false = False
true = True


# Print iterations progress
def printProgressBar (iteration, total, prefix = '', suffix = '', decimals = 1, length = 100, fill = '█', printEnd = "\r"):
	"""
		Call in a loop to create terminal progress bar
		@params:
			iteration   - Required  : current iteration (Int)
			total       - Required  : total iterations (Int)
			prefix      - Optional  : prefix string (Str)
			suffix      - Optional  : suffix string (Str)
			decimals    - Optional  : positive number of decimals in percent complete (Int)
			length      - Optional  : character length of bar (Int)
			fill        - Optional  : bar fill character (Str)
			printEnd    - Optional  : end character (e.g. "\r", "\r\n") (Str)
	"""
	percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
	filledLength = int(length * iteration // total)
	bar = fill * filledLength + '-' * (length - filledLength)
	print(f'\r{prefix} |{bar}| {percent}% {suffix}', end = printEnd)
	# Print New Line on Complete
	if iteration == total: 
		print()


def parse(path):
	g = gzip.open(path, 'rb')
	for l in g:
		yield eval(l)

def getDF(path):
	i = 0
	df = {}
	for d in parse(path):
		df[i] = d
		i += 1
		printProgressBar(i, 9015203)
  
	return pd.DataFrame.from_dict(df, orient='index')

