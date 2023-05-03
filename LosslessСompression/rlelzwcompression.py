import ast
import collections
import math
from matplotlib import pyplot as plt

open("results_rle_lzw.txt", "w")
results = []
N_sequence = 100

def main():
    with open("sequence.txt", "r") as file:
        original_sequences = ast.literal_eval(file.read())
        original_sequences = [sequence.strip("[]',") for sequence in original_sequences]
        file.close()

    for sequence in original_sequences:
        counts = collections.Counter(sequence)
        probability = {symbol: count / N_sequence for symbol, count in counts.items()}
        entropy = -sum(p * math.log2(p) for p in probability.values())

        file = open("results_rle_lzw.txt", "a")
        file.write('Оригінальна послідовність: ' + str(sequence) + '\n')
        file.write('Розмір оригінальної послідовності: ' + str(len(sequence) * 16) + ' bits' + '\n')
        file.write('Ентропія: ' + str(entropy) + '\n')
        file.write('\n')

        file.close()
        encoded_sequence, encoded = encode_rle(sequence)
        decoded_sequence = decode_rle(encoded)
        total_RLE = len(encoded_sequence) * 16
        compression_ratio_RLE = round((len(sequence) / len(encoded_sequence)), 2)
        if compression_ratio_RLE < 1:
            compression_ratio_RLE = '-'
        else:
            compression_ratio_RLE = compression_ratio_RLE
        file = open("results_rle_lzw.txt", "a")
        file.write('___________Кодування_RLE____________' + '\n')
        file.write('Закодована RLE послідовність: ' + str(encoded_sequence) + '\n')
        file.write('Розмір закодованої RLE послідовності: ' + str(total_RLE) + ' bits' + '\n')
        file.write('Коефіцієнт стиснення RLE: ' + str(compression_ratio_RLE) + '\n')
        file.close()
        file = open("results_rle_lzw.txt", "a")
        file.write('Декодована RLE послідовність: ' + str(decoded_sequence) + '\n')
        file.write('Розмір декодованої RLE послідовності: ' + str(len(decoded_sequence) * 16) + ' bits' + '\n')
        file.close()
        with open("results_rle_lzw.txt", "a") as file:
            file.write('___________Кодування_LZW____________' + '\n')
            file.write('_________Поетапне кодування_________' + '\n')
        encoded_result, size = encode_lzw(sequence)
        with open("results_rle_lzw.txt", "a") as file:
            file.write('____________________________________' + '\n')
            file.write(f"Закодована LZW послідовність:{''.join(map(str, encoded_result))} \n")
            file.write(f"Розмір закодованої LZW послідовності: {size} bits \n")
            compression_ratio_LZW = round((len(sequence) * 16 / size), 2)
            if compression_ratio_LZW < 1:
                compression_ratio_LZW = '-'
            else:
                compression_ratio_LZW = compression_ratio_LZW
            file.write(f"Коефіціент стиснення LZW: {compression_ratio_LZW} \n")
            file.close()
        decoded_result_LZW = decode_lzw(encoded_result)
        with open("results_rle_lzw.txt", "a") as file:
            file.write(f"Декодована LZW послідовність:{''.join(map(str, decoded_result_LZW))} \n")
            file.write(f"Розмір декодованої LZW послідовності: {len(decoded_result_LZW) * 16} bits \n ")
            file.write('\n' + '\n' + '\n' + '\n' + '\n')
            file.close()
        results.append([round(entropy, 2), compression_ratio_RLE, compression_ratio_LZW])
    fig, ax = plt.subplots(figsize=(14 / 1.54, 8 / 1.54))
    headers = ['Ентропія', 'КС RLE', 'КС LZW']
    row = ['Послідовність 1', 'Послідовність 2', 'Послідовність 3', 'Послідовність 4', 'Послідовність 5', 'Послідовність 6', 'Послідовність 7', 'Послідовність 8']
    ax.axis('off')
    table = ax.table(cellText=results, colLabels=headers, rowLabels=row, loc='center', cellLoc='center')
    table.set_fontsize(14)
    table.scale(0.8, 2)
    fig.savefig('Результати стиснення методами RLE та LZW' + '.png')

def encode_rle(sequence):
    result = []
    count = 1
    for i, item in enumerate(sequence):
        if i == 0:
            continue
        elif item == sequence[i - 1]:
            count += 1
        else:
            result.append((sequence[i - 1], count))
            count = 1
    result.append((sequence[len(sequence) - 1], count))
    encoded = []
    for i, item in enumerate(result):
        encoded.append(f"{item[1]}{item[0]}")
    return "".join(encoded), result

def decode_rle(sequence):
    result = []
    for item in sequence:
        result.append(item[0] * item[1])
    return "".join(result)

def encode_lzw(sequence):
    dictionary = {}
    for i in range(65536):
        dictionary[chr(i)] = i
    current = ""
    result = []
    size = 0
    for c in sequence:
        new_str = current + c
        if new_str in dictionary:
            current = new_str
        else:
            result.append(dictionary[current])
            dictionary[new_str] = len(dictionary)
            element_bits = 16 if dictionary[current] < 65536 else math.ceil(math.log2(len(dictionary)))
            current = c
            with open("results_rle_lzw.txt", "a") as file:
                file.write(f"Code: {dictionary[current]}, Element: {current}, bits: {element_bits}\n")
                file.close()
            size = size + element_bits
    last = 16 if dictionary[current] < 65536 else math.ceil(math.log2(len(dictionary)))
    size = size + last
    with open("results_rle_lzw.txt", "a") as file:
        file.write(f"Code: {dictionary[current]}, Element: {current}, Bits: {last}\n")
    result.append(dictionary[current])
    return result, size

def decode_lzw(sequence):
    dictionary = {}
    for i in range(65536):
        dictionary[i] = chr(i)
    result = ""
    previous = None
    current = ""
    for code in sequence:
        if code in dictionary:
            current = dictionary[code]
            result += current
            if previous is not None:
                dictionary[len(dictionary)] = previous + current[0]
            previous = current
        else:
            current = previous + previous[0]
            result += current
            dictionary[len(dictionary)] = current
            previous = current
    return result

if __name__ == "__main__":
    main()