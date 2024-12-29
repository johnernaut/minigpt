import torch


def main():
    with open("input.txt", "r", encoding="utf-8") as f:
        text = f.read()

    chars = sorted(list(set(text)))

    # create mapping of chars to ints, then one for ints to chars
    stoi = {ch: i for i, ch in enumerate(chars)}
    itos = {i: ch for i, ch in enumerate(chars)}

    # encode and decode from chars to ints and visa versa
    encode = lambda s: [stoi[c] for c in s]
    decode = lambda l: "".join([itos[i] for i in l])

    data = torch.tensor(encode(text), dtype=torch.long)

    # split the data in to training data and validation data
    # first 90% of data will be training, remaining validation
    n = int(0.9 * len(data))
    train_data = data[:n]
    val_data = data[n:]

    block_size = 8
    x = train_data[:block_size]
    y = train_data[1 : block_size + 1]
    for t in range(block_size):
        context = x[: t + 1]
        target = y[t]
        print(f"when input is {context} the target: {target}")


if __name__ == "__main__":
    main()
