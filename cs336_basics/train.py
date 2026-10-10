import argparse

def train(args):
    pass

def main():
    parser = argparse.ArgumentParser()

    # model
    parser.add_argument("--vocab-size")
    parser.add_argument("--context-length")
    parser.add_argument("--num-layers")
    parser.add_argument("--num-heads")
    parser.add_argument("--emb-dim")
    parser.add_argument("--ffn-dim")

    # data
    parser.add_argument("--train-data")
    parser.add_argument("--val-data")
    parser.add_argument("--checkpoint-output")

    # training
    parser.add_argument("--batch-size")
    parser.add_argument("--num-iters")
    parser.add_argument("--device")
    parser.add_argument("--checkpoint-every")

    # optimizer
    parser.add_argument("--lr")
    parser.add_argument("--weight-decay")
    parser.add_argument("--beta1")
    parser.add_argument("--beta2")
    parser.add_argument("--eps")

    # schedule
    parser.add_argument("--warmup-iters")
    parser.add_argument("--cosine-cycle-iters")
    parser.add_argument("--min-lr")

    # logging
    parser.add_argument("--log-every")
    parser.add_argument("--eval-every")

    args = parser.parse_args()
    train(args)

if __name__ == "__main__":
    main()
