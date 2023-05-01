import random
import string
import collections
import math
from matplotlib import pyplot as plt

def parameters(seq):
    counts = collections.Counter(seq)
    probability = {symbol: count / N_sequence for symbol, count in counts.items()}
    probability_str = ', '.join([f"{symbol}={prob:.4f}" for symbol, prob in probability.items()])
    mean_probability = sum(probability.values()) / len(probability)
    equal = all(abs(prob - mean_probability) < 0.05 * mean_probability for prob in probability.values())
    if equal == mean_probability:
        uniformity = "рівна"
    else:
        uniformity = "нерівна"
    entropy = -sum(p * math.log2(p) for p in probability.values())
    if Sequence_alphabet_size > 1:
        source_excess = 1 - entropy / math.log2(Sequence_alphabet_size)
    else:
        source_excess = 1
    return probability_str, mean_probability, uniformity, entropy, source_excess


# original sequence 1
text = open('results_sequence.txt', 'w')
N_sequence = 100
N1 = 11
list1 = [1]*N1
N0 = N_sequence - N1
list0 = [0]*N0
results = []
sequence = open('sequence.txt', 'r')
original_sequence_1 = list1 + list0
random.shuffle(original_sequence_1)
original_sequence_1 = ''.join(map(str, original_sequence_1))
text.write('Послідовність: ' + str(original_sequence_1) + '\n')
Original_sequence_size = len(original_sequence_1)
text.write('Розмір послідовності: ' + str(Original_sequence_size) + ' byte' + '\n')
unique_chars = set(original_sequence_1)
Sequence_alphabet_size = len(unique_chars)
text.write('Розмір алфавіту: ' + str(Sequence_alphabet_size) + '\n')
probability_str, mean_probability, uniformity, entropy, source_excess = parameters(original_sequence_1)
results.append([Sequence_alphabet_size, round(entropy, 2), round(source_excess, 2), uniformity])
text = open('results_sequence.txt', 'a')
text.write(f'Ймовірність появи символів: {probability_str}\n')
text.write(f'Середнє арифметичне ймовірності: {mean_probability}\n')
text.write(f'Ймовірність розподілу символів: {uniformity}\n')
text.write(f'Ентропія: {entropy}\n')
text.write(f'Надмірність джерела: {source_excess}\n')
text.write('\n')

# original sequence 2
N2 = 5
list2 = ['с', 'к', 'і', 'б', 'а',]
N0_2 = N_sequence - N2
list0_2 = [0]*N0_2
original_sequence_2 = list2+list0_2
original_sequence_2 = ''.join(map(str, original_sequence_2))
text.write('Послідовність: ' + str(original_sequence_2) + '\n')
text.write('Розмір послідовності: ' + str(len(original_sequence_1)) + ' byte' + '\n')
unique_chars_2 = set(original_sequence_2)
Sequence_alphabet_size = len(unique_chars)
text.write('Розмір алфавіту: ' + str(Sequence_alphabet_size) + '\n')
probability_str, mean_probability, uniformity, entropy, source_excess = parameters(original_sequence_2)
results.append([Sequence_alphabet_size, round(entropy, 2), round(source_excess, 2), uniformity])
text = open('results_sequence.txt', 'a')
text.write(f'Ймовірність появи символів: {probability_str}\n')
text.write(f'Середнє арифметичне ймовірності: {mean_probability}\n')
text.write(f'Ймовірність розподілу символів: {uniformity}\n')
text.write(f'Ентропія: {entropy}\n')
text.write(f'Надмірність джерела: {source_excess}\n')
text.write('\n')

# original sequence 3
original_sequence_3 = list(original_sequence_2)
random.shuffle(original_sequence_3)
original_sequence_3 = ''.join(map(str, original_sequence_3))
text.write('Послідовність: ' + str(original_sequence_3) + '\n')
text.write('Розмір послідовності: ' + str(len(original_sequence_3)) + ' byte' + '\n')
unique_chars = set(original_sequence_3)
Sequence_alphabet_size = len(unique_chars)
text.write('Розмір алфавіту: ' + str(Sequence_alphabet_size) + '\n')
probability_str, mean_probability, uniformity, entropy, source_excess = parameters(original_sequence_3)
results.append([Sequence_alphabet_size, round(entropy, 2), round(source_excess, 2), uniformity])
text = open('results_sequence.txt', 'a')
text.write(f'Ймовірність появи символів: {probability_str}\n')
text.write(f'Середнє арифметичне ймовірності: {mean_probability}\n')
text.write(f'Ймовірність розподілу символів: {uniformity}\n')
text.write(f'Ентропія: {entropy}\n')
text.write(f'Надмірність джерела: {source_excess}\n')
text.write('\n')

# original sequence 4
list = []
letters = ['с', 'к', 'і', 'б', 'а', '5', '2', '9']
n_letters = len(letters)
n_repeats = N_sequence/n_letters
remainder = N_sequence * (N_sequence % n_letters)
list += letters * int(n_repeats)
list += letters[:remainder]
original_sequence_4 = ''.join(map(str, list))
text.write('Послідовність: ' + original_sequence_4 + '\n')
text.write('Розмір послідовності: ' + str(len(original_sequence_4)) + ' byte' + '\n')
unique_chars = set(original_sequence_4)
Sequence_alphabet_size = len(unique_chars)
text.write('Розмір алфавіту: ' + str(Sequence_alphabet_size) + '\n')
probability_str, mean_probability, uniformity, entropy, source_excess = parameters(original_sequence_4)
results.append([Sequence_alphabet_size, round(entropy, 2), round(source_excess, 2), uniformity])
text = open('results_sequence.txt', 'a')
text.write(f'Ймовірність появи символів: {probability_str}\n')
text.write(f'Середнє арифметичне ймовірності: {mean_probability}\n')
text.write(f'Ймовірність розподілу символів: {uniformity}\n')
text.write(f'Ентропія: {entropy}\n')
text.write(f'Надмірність джерела: {source_excess}\n')
text.write('\n')

# original sequence 5
alphabet = ['с', 'к', '5', '2', '9']
Pi = 0.2
length = Pi * N_sequence
original_sequence_5 = alphabet * int(length)
random.shuffle(original_sequence_5)
original_sequence_5 = ''.join(map(str, original_sequence_5))
text.write('Послідовність: ' + str(original_sequence_5) + '\n')
text.write('Розмір послідовності: ' + str(len(original_sequence_5)) + ' byte' + '\n')
unique_chars = set(original_sequence_5)
Sequence_alphabet_size = len(unique_chars)
text.write('Розмір алфавіту: ' + str(Sequence_alphabet_size) + '\n')
counts = collections.Counter(original_sequence_5)
probability = {symbol: count / N_sequence for symbol, count in counts.items()}
probability_str = ', '.join([f"{symbol}={prob:.4f}" for symbol, prob in probability.items()])
mean_probability = sum(probability.values()) / len(probability)
equal = all(abs(prob - mean_probability) < 0.05 * mean_probability for prob in probability.values())
if equal:
    uniformity = "рівна"
else:
    uniformity = "нерівна"
entropy = -sum(p * math.log2(p) for p in probability.values())
if Sequence_alphabet_size > 1:
    source_excess = 1 - entropy / math.log2(Sequence_alphabet_size)
else:
    source_excess = 1
results.append([Sequence_alphabet_size, round(entropy, 2), round(source_excess, 2), uniformity])
text = open('results_sequence.txt', 'a')
text.write(f'Ймовірність появи символів: {probability_str}\n')
text.write(f'Середнє арифметичне ймовірності: {mean_probability}\n')
text.write(f'Ймовірність розподілу символів: {uniformity}\n')
text.write(f'Ентропія: {entropy}\n')
text.write(f'Надмірність джерела: {source_excess}\n')
text.write('\n')

# original sequence 6
list_letters = ['с', 'к']
list_digits = ['5', '2', '9']
P_letters = 0.7
P_digits = 0.3
n_letters6 = int(P_letters * N_sequence)/len(list_letters)
n_digits6 = int(P_digits * N_sequence)/len(list_digits)
list_l = list_letters * int(n_letters6)
list_d = list_digits * int(n_digits6)
original_sequence_6 = list_l + list_d
random.shuffle(original_sequence_6)
original_sequence_6 = ''.join(map(str, original_sequence_6))
text.write('Послідовність: ' + str(original_sequence_6) + '\n')
text.write('Розмір послідовності: ' + str(len(original_sequence_6)) + ' byte' + '\n')
unique_chars = set(original_sequence_6)
Sequence_alphabet_size = len(unique_chars)
text.write('Розмір алфавіту: ' + str(Sequence_alphabet_size) + '\n')
probability_str, mean_probability, uniformity, entropy, source_excess = parameters(original_sequence_6)
results.append([Sequence_alphabet_size, round(entropy, 2), round(source_excess, 2), uniformity])
text = open('results_sequence.txt', 'a')
text.write(f'Ймовірність появи символів: {probability_str}\n')
text.write(f'Середнє арифметичне ймовірності: {mean_probability}\n')
text.write(f'Ймовірність розподілу символів: {uniformity}\n')
text.write(f'Ентропія: {entropy}\n')
text.write(f'Надмірність джерела: {source_excess}\n')
text.write('\n')

# original sequence 7
elements = string.ascii_lowercase + string.digits
original_sequence_7 = [random.choice(elements) for _ in range(N_sequence)]
original_sequence_7 = ''.join(map(str, original_sequence_7))
text.write('Послідовність: ' + str(original_sequence_7) + '\n')
text.write('Розмір послідовності: ' + str(len(original_sequence_7)) + ' byte' + '\n')
unique_chars = set(original_sequence_7)
Sequence_alphabet_size = len(unique_chars)
text.write('Розмір алфавіту: ' + str(Sequence_alphabet_size) + '\n')
probability_str, mean_probability, uniformity, entropy, source_excess = parameters(original_sequence_7)
results.append([Sequence_alphabet_size, round(entropy, 2), round(source_excess, 2), uniformity])
text = open('results_sequence.txt', 'a')
text.write(f'Ймовірність появи символів: {probability_str}\n')
text.write(f'Середнє арифметичне ймовірності: {mean_probability}\n')
text.write(f'Ймовірність розподілу символів: {uniformity}\n')
text.write(f'Ентропія: {entropy}\n')
text.write(f'Надмірність джерела: {source_excess}\n')
text.write('\n')

# original sequence 8
original_sequence_8 = ['1'] * N_sequence
original_sequence_8 = ''.join(map(str, original_sequence_8))
text.write('Послідовність: ' + str(original_sequence_8) + '\n')
text.write('Розмір послідовності: ' + str(len(original_sequence_8)) + ' byte' + '\n')
unique_chars = set(original_sequence_8)
Sequence_alphabet_size = len(unique_chars)
text.write('Розмір алфавіту: ' + str(Sequence_alphabet_size) + '\n')
probability_str, mean_probability, uniformity, entropy, source_excess = parameters(original_sequence_8)
results.append([Sequence_alphabet_size, round(entropy, 2), round(source_excess, 2), uniformity])
text = open('results_sequence.txt', 'a')
text.write(f'Ймовірність появи символів: {probability_str}\n')
text.write(f'Середнє арифметичне ймовірності: {mean_probability}\n')
text.write(f'Ймовірність розподілу символів: {uniformity}\n')
text.write(f'Ентропія: {entropy}\n')
text.write(f'Надмірність джерела: {source_excess}\n')
text.write('\n')
text.close()

seq = open('sequence.txt', 'w')
original_sequences = [original_sequence_1, original_sequence_2, original_sequence_3, original_sequence_4, original_sequence_5, original_sequence_6, original_sequence_7, original_sequence_8]
seq.write(str(original_sequences))
seq.close()
text.close()

fig, ax = plt.subplots(figsize=(14/1.54, 8/1.54))
headers = ['Розмір алфавіту', 'Ентропія', 'Надмірність', 'Ймовірність']
row = ['Послідовність 1', 'Послідовність 2', 'Послідовність 3', 'Послідовність 4', 'Послідовність 5', 'Послідовність 6', 'Послідовність 7', 'Послідовність 8']
ax.axis('off')
table = ax.table(cellText=results, colLabels=headers, rowLabels=row, loc='center', cellLoc='center')
table.set_fontsize(14)
table.scale(0.8, 2)
fig.show()
fig.savefig('Характеристики сформованих послідовностей' + '.png')
