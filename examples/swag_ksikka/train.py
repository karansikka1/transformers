import os
import numpy as np
import torch
from datasets import get_swag_data
from transformers import BertTokenizer, AdamW
from modeling_bert import BertForMultipleChoice
import utils
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
from evaluation import evaluate_accuracy
import json

# os.environ["CUDA_VISIBLE_DEVICES"] = "1"


if __name__ == "__main__":
    # Initialize
    args = utils.parse_arguments()
    checkpoint_dir = os.path.join(args.checkpoint_dir, args.model_name)
    result_dir = os.path.join(args.result_dir, args.model_name)
    log_dir = os.path.join(args.logdir, args.model_name)
    utils.create_chkp_result_dirs(checkpoint_dir, result_dir, log_dir, args)
    writer = SummaryWriter(log_dir=log_dir)
    utils.set_random_seed(args.seed)
    wandb = utils.initialize_wandb(args)

    # Multi-GPU
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_ids
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.n_gpu = torch.cuda.device_count()
    args.batch_size = args.batch_size * args.n_gpu
    print("device:", device, "n_gpu:", args.n_gpu, "Batch:size", args.batch_size)
    args.val_freq = int(args.val_freq / args.n_gpu)

    # Model
    knowledge_tuples = {}

    kwargs = {"knowledge_method": args.knowledge_method}
    if args.knowledge_method == 1:
        kwargs["cluster_path"] = "./data/transE_clusters.pkl"

    model = BertForMultipleChoice.from_pretrained("bert-base-uncased", **kwargs)
    text_encoder = BertTokenizer.from_pretrained("bert-base-uncased")
    print("Loaded Model from pretrained")
    model.to(device)
    if args.n_gpu > 1:
        model = torch.nn.DataParallel(model)

    # for param in model.named_parameters():
    #     if "classifier" not in param[0]:
    #         param[1].requires_grad = False
    # Data and dataloaders
    # TODO: Add Attention maps, positinal encoding, segment encoding
    dataset_train, dataset_val = get_swag_data(
        args.data_dir, text_encoder, args.num_validation_samples
    )
    train_loader = torch.utils.data.DataLoader(
        dataset_train,
        batch_size=args.batch_size,
        pin_memory=True,
        num_workers=4,
        shuffle=True,
    )
    val_loader = torch.utils.data.DataLoader(
        dataset_val,
        batch_size=args.batch_size,
        pin_memory=True,
        num_workers=4,
        shuffle=False,
    )

    # Optimizer
    # TODO: Should we use warmup etc. Shift to transformer optimizer that can handle all this
    optimizer = AdamW(model.parameters(), lr=args.lr)
    criterion = torch.nn.CrossEntropyLoss(reduction="sum")
    model.train()

    # Training
    # To keep effective batch-size 32 as per hugging_face examples
    args.gradient_accumulation_steps = int(16 / args.batch_size)
    iternum = 0
    best_perf = -1
    best_iter = 0
    for epoch in range(args.num_epochs):
        for data, label, attention_masks in tqdm(train_loader, desc="Train_Epoch"):
            data = data.to(device)
            label = label.to(device)
            attention_masks = attention_masks.to(device)
            output = model(input_ids=data, attention_mask=attention_masks)
            logits = output[0]
            loss = criterion(logits, label)
            if args.gradient_accumulation_steps > 1:
                loss = loss / args.gradient_accumulation_steps

            loss.backward()

            if iternum % args.gradient_accumulation_steps == 0:
                optimizer.step()
                #     scheduler.step()  # Update learning rate schedule
                optimizer.zero_grad()

            # optimizer.zero_grad()
            # optimizer.step()
            writer.add_scalar("loss", loss.item(), iternum)

            if np.mod(iternum, args.val_freq) == 0:
                acc, scores_test, labels_test = evaluate_accuracy(
                    model, val_loader, device
                )
                tqdm.write(f"Accuracy at {iternum} is {acc:.2f}")
                writer.add_scalar("acc", acc, iternum)
                model.train()
                result_filename = os.path.join(result_dir, f"{iternum}.npy")
                np.save(result_filename, acc)
                if wandb is not None:
                    wandb.run.summary["best_accuracy"] = best_perf
                    wandb.run.summary["best_iter"] = best_iter

                # Save For Debug Info
                """
                f = open(result_filename.replace("npy", "json"), "w")
                prediction = scores_test.argmax(-1)
                sep_token_id = text_encoder.sep_token_id
                for ii in range(100):
                    tmp_data = {}
                    data_tx, label, _ = dataset_val[ii]
                    data_tx = data_tx.numpy()
                    position_sep_token = (
                        (data_tx[0] == sep_token_id).nonzero()[0].tolist()
                    )
                    tmp_data["context"] = text_encoder.decode(
                        data_tx[0, : position_sep_token[0]]
                    )
                    tmp_data["question"] = text_encoder.decode(
                        data_tx[0, position_sep_token[0] + 1 : position_sep_token[1]]
                    )
                    answers = []
                    f or jj in range(4):
                        position_sep_token = (
                            (data_tx[jj] == sep_token_id).nonzero()[0].tolist()
                        )
                        answers.append(
                            text_encoder.decode(
                                data_tx[jj][
                                    position_sep_token[1] + 1 : position_sep_token[2]
                                ]
                            )
                        )
                    tmp_data["answers"] = answers
                    tmp_data["gnd_label"] = [label.item()]
                    tmp_data["pred_label"] = [prediction[ii].item()]
                    json.dump(tmp_data, f, indent=True)

                f.close()
                """

                if acc > best_perf:
                    best_perf = acc
                    best_iter = iternum
                    if args.save:
                        checkpoint = {
                            "state_dict": model.state_dict,
                            "args": args,
                            "best_iter": best_iter,
                            "best_perf": best_perf,
                        }
                        torch.save(checkpoint, os.path.join(checkpoint_dir, "best.pt"))

            iternum += 1

        print("Finish Epoch")
