#!/usr/bin/env python3
import pickle


def main():
    vocab = dict()
    with open("resources/vocab_cut.txt") as f:
        vocab = {line.strip(): idx for idx, line in enumerate(f)}

    with open("resources/vocab.pkl", "wb") as f:
        pickle.dump(vocab, f, pickle.HIGHEST_PROTOCOL)

if __name__ == "__main__":
    main()

