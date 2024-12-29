def main():
    with open("input.txt", "r", encoding="utf-8") as f:
        text = f.read()

    chars = sorted(list(set(text)))
    vocab_size = len(chars)
    print(vocab_size)

if __name__ == "__main__":
    main()
