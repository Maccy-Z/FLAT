import torch
import time
import itertools
import numpy as np
from flat_model import ModelHolder
from config import get_config
from dataloader import d2v_pairer, SplitDataloader
from save_holder import SaveHolder


def main(all_cfgs, nametag=None):
    save_holder = None

    cfg = all_cfgs["DL_params"]
    bs = cfg["bs"]
    num_rows = cfg["num_rows"]
    num_targets = cfg["num_targets"]
    fold_no = cfg["fold_no"]
    num_cols = cfg.get("num_cols")
    decrease_col_prob = cfg.get("decrease_col_prob")
    num_1s = cfg.get("num_1s")
    binarise = cfg["binarise"]
    folds_file = cfg.get("folds_file")

    cfg = all_cfgs["Settings"]
    num_epochs = cfg["num_epochs"]

    val_interval = cfg["val_interval"]
    val_duration = cfg["val_duration"]

    folds_file = f"./datasets/grouped_datasets/{folds_file}"
    dl = SplitDataloader(
        bs=bs, num_rows=num_rows, num_targets=num_targets,
        binarise=binarise,
        num_1s=num_1s,
        num_cols=num_cols['train'],
        decrease_col_prob=decrease_col_prob,
        fold_no=fold_no, ds_split="train",
        folds_file=folds_file
    )
    print("Training data names:", dl.all_datasets)

    # val_dl =  SplitDataloader() # optional validation loader
    val_dl = None

    cfg = all_cfgs["Optim"]
    lr = cfg["lr"]
    eps = cfg["eps"]
    decay = cfg["decay"]

    model = ModelHolder(cfg_all=all_cfgs)

    optim = torch.optim.AdamW(model.parameters(), lr=lr, eps=eps, weight_decay=decay)

    accs, losses = [], []
    val_accs, val_losses = [], []
    st = time.time()
    for epoch in range(num_epochs):
        duration = time.time() - st
        st = time.time()
        print()
        print(f'{epoch = }, {duration = :.2g}s')

        # Train loop
        model.train()
        for xs, ys, _ in itertools.islice(dl, val_interval):
            # Train loop
            # xs.shape = [bs, num_rows+num_targets, num_cols]
            xs_meta, xs_target = xs[:, :num_rows], xs[:, num_rows:]
            ys_meta, ys_target = ys[:, :num_rows], ys[:, num_rows:]
            # Splicing like this changes the tensor's stride. Fix here:
            xs_meta, xs_target = xs_meta.contiguous(), xs_target.contiguous()
            ys_meta, ys_target = ys_meta.contiguous(), ys_target.contiguous()
            ys_target = ys_target.view(-1)

            # Reshape for dataset2vec
            pairs_meta = d2v_pairer(xs_meta, ys_meta)
            # First pass with the meta-set, train d2v and get embedding.

            embed_meta, pos_enc = model.forward_meta(pairs_meta)
            # Second pass using previous embedding and train weight encoder
            # During testing, rows of the dataset don't interact.
            ys_pred_targ = model.forward_target(xs_target, embed_meta, pos_enc).view(-1, 2)

            loss = model.loss_fn(ys_pred_targ, ys_target)
            loss.backward()

            optim.step()
            optim.zero_grad()

            # Accuracy recording
            predicted_labels = torch.argmax(ys_pred_targ, dim=1)
            accuracy = (predicted_labels == ys_target).sum().item() / len(ys_target)

            accs.append(accuracy), losses.append(loss.item())

        print(f"Training accuracy : {np.mean(accs[-val_interval:]) * 100:.2f}%")

        if val_dl:
            # Validation loop
            model.eval()
            epoch_accs, epoch_losses = [], []
            save_ys_targ, save_pred_labs = [], []
            for xs, ys, _ in itertools.islice(val_dl, val_duration):
                # xs.shape = [bs, num_rows+1, num_cols]
                xs_meta, xs_target = xs[:, :num_rows], xs[:, num_rows:]
                ys_meta, ys_target = ys[:, :num_rows], ys[:, num_rows:]
                # Splicing like this changes the tensor's stride. Fix here:
                xs_meta, xs_target = xs_meta.contiguous(), xs_target.contiguous()
                ys_meta, ys_target = ys_meta.contiguous(), ys_target.contiguous()
                ys_target = ys_target.view(-1)

                # Reshape for dataset2vec
                pairs_meta = d2v_pairer(xs_meta, ys_meta)

                with torch.no_grad():
                    embed_meta, pos_enc = model.forward_meta(pairs_meta)
                    ys_pred_targ = model.forward_target(xs_target, embed_meta, pos_enc).view(-1, 2)
                    loss = torch.nn.functional.cross_entropy(ys_pred_targ, ys_target.long())

                # Accuracy recording
                predicted_labels = torch.argmax(ys_pred_targ, dim=1)
                accuracy = (predicted_labels == ys_target).sum().item() / len(ys_target)

                epoch_accs.append(accuracy), epoch_losses.append(loss.item())
                save_ys_targ.append(ys_target)
                save_pred_labs.append(predicted_labels)

            val_losses.append(epoch_losses), val_accs.append(epoch_accs)

            print(f'Validation accuracy: {np.mean(val_accs[-1]) * 100:.2f}%')
        
        # Save stats
        if save_holder is None:
            save_holder = SaveHolder(".", nametag=nametag)
        history = {"accs": accs, "loss": losses, "val_accs": val_accs, "val_loss": val_losses, "epoch_no": epoch}
        save_holder.save_model(model, optim, epoch=epoch)
        save_holder.save_history(history)


if __name__ == "__main__":
    print("Training starting")

    dev = torch.device("cpu")

    main(all_cfgs=get_config())

    print("")
    print("Training Completed")