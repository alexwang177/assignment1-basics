import argparse
import logging
import numpy as np

from cs336_basics.utils import *
from cs336_basics.optimizer import *
from cs336_basics.lr_scheduler import *
from cs336_basics.data import *
from cs336_basics.loss import *

logging.basicConfig(                                        
    level=logging.INFO,                                     
    format="%(asctime)s | %(levelname)s | %(message)s",     
) 

def train(args):
    
    # initialize model
    model = TransformerLM(
        vocab_size=args.vocab_size,
        context_length=args.context_length,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        d_model=args.emb_dim,
        d_ff=args.ffn_dim,
        rope_theta=args.rope_theta
    )
    model = model.to(args.device)

    logging.info("Initializing model")
    logging.info(model)

    # initialize optimizer
    logging.info("Initializing optimizer")
    optimizer = AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
        betas=(args.beta1, args.beta2),
        eps=args.eps
    )

    # prepare dataset
    train_dataset = np.load(args.train_data, mmap_mode="r")

    # training loop
    model.train()
    for it in range(args.num_iters):

        inputs, targets = get_batch(
            dataset=train_dataset,
            batch_size=args.batch_size,
            context_length=args.context_length,
            device=args.device
        )

        optimizer.zero_grad()
        logits = model(inputs)
        loss = optimized_ce_loss(logits, targets)
        loss.backward()

        # apply lr schedule
        lr = lr_cosine_scheduler(                                            
            it=it,                                                       
            max_learning_rate=args.lr,                                          
            min_learning_rate=args.min_lr,                                      
            warmup_iters=args.warmup_iters,                                     
            cosine_cycle_iters=args.cosine_cycle_iters,                         
        )     
        for group in optimizer.param_groups:
            group["lr"] = lr

        optimizer.step()

        if it % args.log_every == 0:
            logging.info(f"Iteration {it}, lr={lr}, loss={loss.item()}")


def main():
    parser = argparse.ArgumentParser()

    # model
    parser.add_argument("--vocab-size", type=int, default=128_256)
    parser.add_argument("--context-length", type=int, default=256)
    parser.add_argument("--num-layers", type=int, default=4)
    parser.add_argument("--num-heads", type=int, default=4)
    parser.add_argument("--emb-dim", type=int, default=256)
    parser.add_argument("--ffn-dim", type=int, default=1024)
    parser.add_argument("--rope-theta", type=float, default=10000.0) 

    # data
    parser.add_argument("--train-data", required=True)
    parser.add_argument("--val-data")
    parser.add_argument("--checkpoint-output")

    # training
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-iters", type=int, default=1000)
    parser.add_argument("--device", default="mps")
    parser.add_argument("--checkpoint-every", type=int, default=10)

    # optimizer
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--beta1", type=float, default=0.9)
    parser.add_argument("--beta2", type=float, default=0.95)
    parser.add_argument("--eps", type=float, default=1e-8)

    # schedule
    parser.add_argument("--warmup-iters", type=int, default=100)
    parser.add_argument("--cosine-cycle-iters", type=int, default=1000)
    parser.add_argument("--min-lr", type=float, default=3e-5)

    # logging
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--eval-every")

    args = parser.parse_args()
    train(args)

if __name__ == "__main__":
    main()
