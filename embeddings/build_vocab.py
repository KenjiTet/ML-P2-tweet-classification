from collections import Counter
import re

def build_vocab(filenames, output_file):
    word_count = Counter()
    for filename in filenames:
        with open(filename, 'r', encoding='utf-8') as file:
            for line in file:
                # Splitting the line into words and counting
                words = re.findall(r'\w+', line.lower())
                word_count.update(words)
    
    with open(output_file, 'w', encoding='utf-8') as file:
        for word, count in word_count.items():
            file.write(f'{count} {word}\n')



def cut_vocab(input_file, output_file, min_count=5):
    with open(input_file, 'r', encoding='utf-8') as file:
        vocab = [line for line in file if int(line.split()[0]) >= min_count]

    with open(output_file, 'w', encoding='utf-8') as file:
        for line in vocab:
            word = line.split()[1]
            file.write(f'{word}\n')


def main():
    input_file_path1 = "twitter-datasets/prep_small_pos.txt"
    input_file_path2 = "twitter-datasets/prep_small_neg.txt"

    build_vocab([input_file_path1, input_file_path2], 'resources/vocab_full.txt')
    cut_vocab('resources/vocab_full.txt', 'resources/vocab_cut.txt')

if __name__ == "__main__":
    main()
