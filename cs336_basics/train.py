import argparse
import logging
import numpy as np
import torch
import random
import time

from cs336_basics.utils import *
from cs336_basics.optimizer import *
from cs336_basics.lr_scheduler import *
from cs336_basics.data import *
from cs336_basics.loss import *
from cs336_basics.checkpoint import *

logging.basicConfig(                                        
    level=logging.INFO,                                     
    format="%(asctime)s | %(levelname)s | %(message)s",     
) 

def evaluate(
    args,
    model,
    val_dataset,
):
    logging.info("Starting evaluation...")
    model.eval()

    eval_loss = 0

    with torch.no_grad():

        for _ in range(args.eval_iters):

            inputs, targets = get_batch(
                dataset=val_dataset,
                batch_size=args.batch_size,
                context_length=args.context_length,
                device=args.device
            )

            logits = model(inputs)
            loss = optimized_ce_loss(logits, targets)
            eval_loss += loss.item()

    eval_loss /= args.eval_iters
    logging.info(f"Validation loss={eval_loss}")

    model.train()


def train(args):

    # Set seeds for randomness
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    
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

    start_iteration = 0

    # resume from checkpoint
    if args.resume_from is not None:
        start_iteration = load_checkpoint(
            src=args.resume_from,
            model=model,
            optimizer=optimizer
        )

    # prepare datasets
    train_dataset = np.load(args.train_data, mmap_mode="r")
    val_dataset = None

    if args.val_data is not None:
        if args.eval_every is None:
            raise Exception("eval-every must be set if val-data is set.")

        val_dataset = np.load(args.val_data, mmap_mode="r")

    # check dummy batch shape
    inputs, targets = get_batch(
        dataset=train_dataset,
        batch_size=args.batch_size,
        context_length=args.context_length,
        device=args.device
    )
    logging.info(f"inputs shape = {inputs.shape}")
    logging.info(f"targets shape = {targets.shape}")

    # training loop
    model.train()

    for it in range(start_iteration, start_iteration + args.num_iters):

        TRAIN_STEP_START_TS = time.time()

        inputs, targets = get_batch(
            dataset=train_dataset,
            batch_size=args.batch_size,
            context_length=args.context_length,
            device=args.device
        )

        optimizer.zero_grad()
        logits = model(inputs)
        loss = optimized_ce_loss(logits, targets)

        if not torch.isfinite(loss):
            raise Exception(f"Invalid loss value: {loss.item()}")

        loss.backward()

        for name, param in model.named_parameters():
            if param.grad is not None and not torch.isfinite(param.grad).all().item():
                raise Exception(f"Invalid gradient value for {name}")

        # apply grad clip
        grad_clip(
            parameters=model.parameters(), 
            max_l2_norm=args.max_grad_norm
        )

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

        if args.device == "mps" and hasattr(torch, "mps"):       
            torch.mps.synchronize()
        TRAIN_STEP_END_TS = time.time()
        TRAIN_STEP_ELAPSED_SEC = TRAIN_STEP_END_TS - TRAIN_STEP_START_TS
        tokens_per_sec = (args.batch_size * args.context_length) / TRAIN_STEP_ELAPSED_SEC

        if (it + 1) % args.log_every == 0:
            logging.info(f"Iteration {it + 1}, lr={lr}, loss={loss.item()}, tokens/sec={tokens_per_sec}")

        if args.val_data is not None and args.eval_every is not None and (it + 1) % args.eval_every == 0:
            evaluate(args, model, val_dataset)

        if (it + 1) % args.checkpoint_every == 0 and args.checkpoint_output is not None:
            logging.info(f"Iteration {it + 1} Saving checkpoint...")
            save_checkpoint(
                model=model,
                optimizer=optimizer,
                iteration=it + 1,
                out=f"{args.checkpoint_output}/it={it+1}"
            )


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

    # checkpointing
    parser.add_argument("--checkpoint-output")
    parser.add_argument("--checkpoint-every", type=int, default=10)
    parser.add_argument("--resume-from")

    # evaluation
    parser.add_argument("--eval-every", type=int)
    parser.add_argument("--eval-iters", type=int, default=10)

    # training
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--num-iters", type=int, default=1000)
    parser.add_argument("--device", default="mps")
    parser.add_argument("--max-grad-norm", type=float, default=1.0)

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

    args = parser.parse_args()
    train(args)

if __name__ == "__main__":
    main()
