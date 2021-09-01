import pandas as pd


def getMaxColumnFile(file, column):
    return pd.read_csv(file, sep=',')[column].max()


print(getMaxColumnFile('Output/neural_sort_cross_entropy_default/log.txt', 'NDCG10'))
print(getMaxColumnFile('Output/neural_sort_cross_entropy_multdrop/logmultipledrop.txt', 'NDCG10'))
print("--")
print(getMaxColumnFile('Output/neural_sort_cross_entropy_default/log.txt', 'NDCG5'))
print(getMaxColumnFile('Output/neural_sort_cross_entropy_multdrop/logmultipledrop.txt', 'NDCG5'))
print("--")
print(getMaxColumnFile('Output/neural_sort_cross_entropy_default/log.txt', 'HitRatio10'))
print(getMaxColumnFile('Output/neural_sort_cross_entropy_multdrop/logmultipledrop.txt', 'HitRatio10'))