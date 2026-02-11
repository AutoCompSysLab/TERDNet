"""
TERDNet training script.

This codebase was originally based on:
  https://github.com/DoctorKey/C-3PO

Refactor notes (TERDNet):
- Default model name is now: vit_b_terdnet
- Optimizer param groups are selected by checking model attributes
  (backbone/classifier) rather than parsing model name strings.
- CLI keeps backward compatibility:
    --mtf == --fusion-type
    --msf == --pyramid-levels
    --iter == --decoder-iters
"""

import datetime
import os
import time
import math
from collections import OrderedDict

import numpy as np
import torch
import torch.utils.data

import utils
from dataset import dataset_dict
from model import model_dict
import loss

def get_scheduler_function(name, total_iters, final_lr=0.0):
    print(f"LR Scheduler: {name}")
    if name == "cosine":
        return lambda step: ((1 + math.cos(step * math.pi / total_iters)) / 2) * (1 - final_lr) + final_lr
    elif name == "linear":
        return lambda step: 1 - (1 - final_lr) / total_iters * step
    elif name == "exp":
        return lambda step: (1 - step / total_iters) ** 0.9
    elif name == "none":
        return lambda step: 1.0
    else:
        raise ValueError(name)


def warmup(num_iter, num_warmup, optimizer):
    """
    Linear LR warmup from 0.0 to each param_group's configured LR.
    """
    if num_iter < num_warmup:
        xi = [0, num_warmup]
        for j, group in enumerate(optimizer.param_groups):
            group["lr"] = np.interp(num_iter, xi, [0.0, group["lr"]])
            if "momentum" in group:
                group["momentum"] = np.interp(num_iter, xi, [0.8, 0.9])


def fix_BN_stat(module):
    classname = module.__class__.__name__
    if classname.find("BatchNorm2d") != -1:
        module.eval()


def freeze_BN_stat(model):
    """
    Keep original behavior: only apply in DDP case, and only to model.module.backbone
    if it exists. (Useful for some backbones / small batch sizes.)
    """
    if isinstance(model, torch.nn.parallel.DistributedDataParallel):
        if hasattr(model.module, "backbone"):
            print("freeze backbone BN stat")
            model.module.backbone.apply(fix_BN_stat)


def CD_evaluate(model, data_loader, iter, device, save_imgs_dir=None):
    model.eval()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter("Prec", utils.SmoothedValue(window_size=1, fmt="{value:.3f} ({global_avg:.3f})"))
    metric_logger.add_meter("Rec", utils.SmoothedValue(window_size=1, fmt="{value:.3f} ({global_avg:.3f})"))
    metric_logger.add_meter("Acc", utils.SmoothedValue(window_size=1, fmt="{value:.3f} ({global_avg:.3f})"))
    metric_logger.add_meter("F1score", utils.SmoothedValue(window_size=1, fmt="{value:.4f} ({global_avg:.4f})"))
    header = "Test:"
    with torch.no_grad():
        for image, target in metric_logger.log_every(data_loader, 100, header):
            image, target = image.to(device), target.to(device)
            output = model(image)
            if isinstance(output, OrderedDict):
                output = output["out"]
                output = output[int(iter - 1)]
            mask_pred = torch.topk(output.data, 1, dim=1)[1][:, 0]
            mask_gt = (target > 0)[:, 0]
            precision, recall, accuracy, f1score = utils.CD_metric_torch(mask_pred, mask_gt)

            metric_logger.Prec.update(precision.mean(), n=len(precision))
            metric_logger.Rec.update(recall.mean(), n=len(precision))
            metric_logger.Acc.update(accuracy.mean(), n=len(precision))
            metric_logger.F1score.update(f1score.mean(), n=len(f1score))

            if save_imgs_dir:
                assert len(precision) == 1, "save imgs needs batch_size=1"
                output_pil = data_loader.dataset.get_pil(image[0], mask_gt, mask_pred)
                output_pil.save(
                    os.path.join(save_imgs_dir, f"{utils.get_rank()}_{metric_logger.F1score.count}.png")
                )

        metric_logger.synchronize_between_processes()

    print(
        "{} {} Total: {} Metric Prec: {:.4f} Recall: {:.4f} F1: {:.4f}".format(
            header,
            data_loader.dataset.name,
            metric_logger.F1score.count,
            metric_logger.Prec.global_avg,
            metric_logger.Rec.global_avg,
            metric_logger.F1score.global_avg,
        )
    )
    return metric_logger.F1score.global_avg


def SS_evaluate(model, data_loader, iter, device, save_imgs_dir=None):
    model.eval()
    num_classes = data_loader.dataset.num_classes
    if hasattr(data_loader.dataset, "class_mask"):
        class_mask = data_loader.dataset.class_mask
    else:
        class_mask = None

    confmat = utils.ConfusionMatrix(num_classes, class_mask)
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = "Test:"
    count = 0
    with torch.no_grad():
        for image, target in metric_logger.log_every(data_loader, 100, header):
            image, target = image.to(device), target.to(device)
            output = model(image)
            if isinstance(output, OrderedDict):
                output = output["out"]
                output = output[int(iter - 1)]
            pred = output.argmax(1)
            confmat.update(target.flatten(), pred.flatten())

            if save_imgs_dir:
                if count > 100:
                    continue
                output_pil = data_loader.dataset.get_pil(image[0], target[0], pred[0])
                output_pil.save(os.path.join(save_imgs_dir, f"{utils.get_rank()}_{count}.png"))
                count += 1

        confmat.reduce_from_all_processes()

    print(f"{header} {data_loader.dataset.name} confmat: {confmat}")
    acc_global, acc, iu = confmat.compute()
    mIoU = confmat.mIoU(iu)
    return mIoU


def CD_train_one_epoch(
    model, criterion, optimizer, scaler, data_loader, lr_scheduler, num_warmup, device, epoch, print_freq, iter
):
    model.train()
    freeze_BN_stat(model)
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter("lr", utils.SmoothedValue(window_size=1, fmt="{value}"))
    metric_logger.add_meter("f1score", utils.SmoothedValue(window_size=1, fmt="{value:.4f} ({global_avg:.4f})"))
    header = f"Epoch: [{epoch}]"

    warmup(lr_scheduler._step_count, num_warmup, optimizer)

    for image, target in metric_logger.log_every(data_loader, print_freq, header):
        image, target = image.to(device), target.to(device)

        output = model(image)
        loss_val = criterion(output, target[:, 0])

        optimizer.zero_grad()
        scaler.scale(loss_val).backward()
        scaler.step(optimizer)
        scaler.update()

        lr_scheduler.step()
        warmup(lr_scheduler._step_count, num_warmup, optimizer)

        out_for_metric = output
        if isinstance(out_for_metric, OrderedDict):
            out_for_metric = out_for_metric["out"]
            out_for_metric = out_for_metric[int(iter - 1)]

        mask_pred = torch.topk(out_for_metric.data, 1, dim=1)[1][:, 0]
        mask_gt = (target > 0)[:, 0]
        _, _, _, f1score = utils.CD_metric_torch(mask_pred, mask_gt)

        metric_logger.update(loss=loss_val.item(), lr=optimizer.param_groups[0]["lr"])
        metric_logger.f1score.update(f1score.mean(), n=len(f1score))


def SS_train_one_epoch(
    model, criterion, optimizer, scaler, data_loader, lr_scheduler, num_warmup, device, epoch, print_freq, iter
):
    model.train()
    freeze_BN_stat(model)
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter("lr", utils.SmoothedValue(window_size=1, fmt="{value}"))
    header = f"Epoch: [{epoch}]"

    warmup(lr_scheduler._step_count, num_warmup, optimizer)

    for image, target in metric_logger.log_every(data_loader, print_freq, header):
        image, target = image.to(device), target.to(device)
        output = model(image)
        target = target.squeeze()
        loss_val = criterion(output, target)

        optimizer.zero_grad()
        loss_val.backward()
        optimizer.step()

        lr_scheduler.step()
        warmup(lr_scheduler._step_count, num_warmup, optimizer)

        metric_logger.update(loss=loss_val.item(), lr=optimizer.param_groups[0]["lr"])


def create_dataloader(args):
    dataset = dataset_dict[args.train_dataset](args, train=True)
    dataset_test = dataset_dict[args.test_dataset](args, train=False)

    if args.test_dataset2:
        dataset_test2 = dataset_dict[args.test_dataset2](args, train=False)
    else:
        dataset_test2 = None

    if args.distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(dataset)
        test_sampler = torch.utils.data.distributed.DistributedSampler(dataset_test)
        test_sampler2 = torch.utils.data.distributed.DistributedSampler(dataset_test2) if dataset_test2 else None
    else:
        train_sampler = torch.utils.data.RandomSampler(dataset)
        test_sampler = torch.utils.data.SequentialSampler(dataset_test)
        test_sampler2 = torch.utils.data.SequentialSampler(dataset_test2) if dataset_test2 else None

    data_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        sampler=train_sampler,
        num_workers=args.workers,
        collate_fn=utils.collate_fn,
        drop_last=True,
    )

    data_loader_test = torch.utils.data.DataLoader(
        dataset_test,
        batch_size=1,
        sampler=test_sampler,
        num_workers=args.workers,
        collate_fn=utils.collate_fn,
    )

    if dataset_test2:
        data_loader_test2 = torch.utils.data.DataLoader(
            dataset_test2,
            batch_size=1,
            sampler=test_sampler2,
            num_workers=args.workers,
            collate_fn=utils.collate_fn,
        )
    else:
        data_loader_test2 = None

    return dataset, train_sampler, data_loader, dataset_test, data_loader_test, dataset_test2, data_loader_test2


def prepare_train(args, model_without_ddp, dataset, data_loader):
    """
    Params selection:
    - If model exposes .backbone and .classifier, optimize them separately.
      (Works for vit_b_terdnet / any TERDNetModel wrapper.)
    - Otherwise, optimize all trainable parameters.
    """
    if hasattr(model_without_ddp, "backbone") and hasattr(model_without_ddp, "classifier"):
        params_to_optimize = [
            {"params": [p for p in model_without_ddp.backbone.parameters() if p.requires_grad]},
            {"params": [p for p in model_without_ddp.classifier.parameters() if p.requires_grad]},
        ]
    else:
        params_to_optimize = [p for p in model_without_ddp.parameters() if p.requires_grad]

    if args.opt == "adam":
        optimizer = torch.optim.Adam(
            params_to_optimize,
            lr=args.lr,
            betas=(args.momentum, 0.999),
            weight_decay=args.weight_decay,
        )
    elif args.opt == "sgd":
        optimizer = torch.optim.SGD(
            params_to_optimize,
            lr=args.lr,
            momentum=args.momentum,
            weight_decay=args.weight_decay,
        )
    else:
        raise ValueError(args.opt)

    if args.loss_weight:
        print("computing loss weight")
        ratio = dataset.get_mask_ratio()
        loss_weight = torch.tensor(ratio).cuda()
        print(f"loss weight {loss_weight}")
    else:
        loss_weight = None

    criterion = loss.get_loss(args.loss, loss_weight)
    lambda_lr = get_scheduler_function(args.lr_scheduler, args.epochs * len(data_loader), final_lr=0.2 * args.lr)
    lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_lr)

    if args.warmup:
        num_warmup = max(round(5 * len(data_loader)), 1000)
    else:
        num_warmup = 0

    scaler = torch.cuda.amp.GradScaler(enabled=True)
    return optimizer, criterion, lr_scheduler, scaler, num_warmup


def main(args):
    if args.output_dir:
        utils.mkdir(args.output_dir)

    utils.init_distributed_mode(args)
    print(args)

    device = torch.device(args.device)

    dataset, train_sampler, data_loader, dataset_test, data_loader_test, dataset_test2, data_loader_test2 = (
        create_dataloader(args)
    )

    # Choose task loop
    if args.train_dataset in ["COCO", "ChangeSim_Multi", "ChangeSim_Binary", "ChangeSim_Semantic", "CSim", "PSCD"]:
        train_one_epoch = SS_train_one_epoch
        evaluate = SS_evaluate
    else:
        train_one_epoch = CD_train_one_epoch
        evaluate = CD_evaluate

    args.num_classes = dataset.num_classes

    # Backward-compat: if old model name is passed, map to new.
    if args.model == "vit_b_mtf_msf_deeplabv3":
        print("[DEPRECATION] --model vit_b_mtf_msf_deeplabv3 is deprecated. Use --model vit_b_terdnet.")
        args.model = "vit_b_terdnet"

    if args.model not in model_dict:
        raise KeyError(
            f"Unknown model '{args.model}'. Available keys: {list(model_dict.keys())[:20]}..."
        )

    model = model_dict[args.model](args)
    model.to(device)

    if args.distributed:
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)

    model_without_ddp = model
    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu], find_unused_parameters=True)
        model_without_ddp = model.module

    optimizer, criterion, lr_scheduler, scaler, num_warmup = prepare_train(args, model_without_ddp, dataset, data_loader)

    if args.pretrained:
        utils.load_model(model_without_ddp, args.pretrained)

    if args.resume:
        print(f"load from: {args.resume}")
        checkpoint = torch.load(args.resume, map_location="cpu")
        sd = checkpoint["model"]
        ret = model_without_ddp.load_state_dict(sd, strict=not args.test_only)
        print(f"load ret: {ret}")
        if not args.test_only:
            optimizer.load_state_dict(checkpoint["optimizer"])
            lr_scheduler.load_state_dict(checkpoint["lr_scheduler"])
            args.start_epoch = checkpoint["epoch"] + 1

    if args.test_only:
        if args.save_imgs:
            save_imgs_dir = os.path.join(args.output_dir, "img")
            os.makedirs(save_imgs_dir, exist_ok=True)
        else:
            save_imgs_dir = None

        score = evaluate(model, data_loader_test, iter=args.iter, device=device, save_imgs_dir=save_imgs_dir)
        print(score)
        return

    best = -1.0
    start_time = time.time()

    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            train_sampler.set_epoch(epoch)

        train_one_epoch(
            model,
            criterion,
            optimizer,
            scaler,
            data_loader,
            lr_scheduler,
            num_warmup,
            device,
            epoch,
            args.print_freq,
            args.iter,
        )

        if epoch % args.eval_every == 0:
            score = evaluate(model, data_loader_test, iter=args.iter, device=device)
            if dataset_test2:
                _ = evaluate(model, data_loader_test2, iter=args.iter, device=device)
        else:
            score = -1.0

        checkpoint = {
            "model": model_without_ddp.state_dict(),
            "optimizer": optimizer.state_dict(),
            "lr_scheduler": lr_scheduler.state_dict(),
            "epoch": epoch,
            "args": args,
        }

        if score > best:
            best = score
            utils.save_on_master(checkpoint, os.path.join(args.output_dir, "best.pth"))

        utils.save_on_master(checkpoint, os.path.join(args.output_dir, "checkpoint.pth"))

    # Final evaluation + save
    if args.train_dataset in ["COCO"]:
        save_imgs_dir = None
    else:
        save_imgs_dir = os.path.join(args.output_dir, f"{dataset_test.name}_img")
        os.makedirs(save_imgs_dir, exist_ok=True)

    _ = evaluate(model, data_loader_test, iter=args.iter, device=device, save_imgs_dir=save_imgs_dir)

    if dataset_test2:
        save_imgs_dir2 = os.path.join(args.output_dir, f"{dataset_test2.name}_img")
        os.makedirs(save_imgs_dir2, exist_ok=True)
        _ = evaluate(model, data_loader_test2, iter=args.iter, device=device, save_imgs_dir=save_imgs_dir2)

    utils.save_on_master(checkpoint, os.path.join(args.output_dir, f"model_{epoch}.pth"))

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print(f"Training time {total_time_str}")


def get_args_parser(add_help=True):
    import argparse

    parser = argparse.ArgumentParser(description="PyTorch Scene Change Detection (TERDNet)", add_help=add_help)

    parser.add_argument("--train-dataset", default="VL_CMU_CD", help="dataset name")
    parser.add_argument("--test-dataset", default="VL_CMU_CD", help="dataset name")
    parser.add_argument("--test-dataset2", default="", help="dataset name")

    parser.add_argument("--input-size", default=448, type=int, metavar="N", help="the input-size of images")
    parser.add_argument("--randomflip", default=0.5, type=float, help="random flip input")
    parser.add_argument("--randomrotate", dest="randomrotate", action="store_true", help="random rotate input")
    parser.add_argument("--randomcrop", dest="randomcrop", action="store_true", help="random crop input")
    parser.add_argument("--data-cv", default=0, type=int, metavar="N", help="the number of cross validation")

    # Model / TERDNet options
    parser.add_argument("--model", default="vit_b_terdnet", help="model name (e.g., vit_b_terdnet)")

    # Keep legacy flag names but provide paper-friendly aliases.
    parser.add_argument(
        "--mtf",
        "--fusion-type",
        dest="mtf",
        default="ours",
        type=str,
        help=(
            "Feature fusion type. "
            "Recommended: ours | feature | local_corr | global_corr. "
            "Legacy aliases also accepted: corrconv | conv | corr | globalcorr."
        ),
    )
    parser.add_argument(
        "--msf",
        "--pyramid-levels",
        dest="msf",
        default=4,
        type=int,
        help="number of feature pyramid levels (TERDNet uses 4)",
    )
    parser.add_argument(
        "--iter",
        "--decoder-iters",
        dest="iter",
        default=5,
        type=int,
        help="number of recurrent refinement iterations",
    )
    parser.add_argument(
        "--sam-checkpoint",
        default="",
        type=str,
        help="optional SAM checkpoint path (if empty, use default ./checkpoints/...)",
    )

    # Train settings
    parser.add_argument("--device", default="cuda", help="device")
    parser.add_argument("-b", "--batch-size", default=4, type=int)
    parser.add_argument("--epochs", default=100, type=int, metavar="N", help="number of total epochs to run")

    parser.add_argument(
        "-j", "--workers", default=16, type=int, metavar="N", help="number of data loading workers (default: 16)"
    )
    parser.add_argument("--loss", default="bi", type=str, help="the training loss")
    parser.add_argument("--loss-weight", action="store_true", help="add weight for loss")
    parser.add_argument("--opt", default="adam", type=str, help="the optimizer")
    parser.add_argument("--lr-scheduler", default="cosine", type=str, help="the lr scheduler")
    parser.add_argument("--lr", default=0.0001, type=float, help="initial learning rate")
    parser.add_argument("--warmup", dest="warmup", action="store_true", help="warmup the lr")
    parser.add_argument("--momentum", default=0.9, type=float, metavar="M", help="momentum")
    parser.add_argument("--wd", "--weight-decay", default=0.0, type=float, metavar="W", help="weight decay", dest="weight_decay")
    parser.add_argument("--print-freq", default=50, type=int, help="print frequency")
    parser.add_argument("--resume", default="", help="resume from checkpoint")
    parser.add_argument("--pretrained", default="", help="pretrain checkpoint")
    parser.add_argument("--start-epoch", default=0, type=int, metavar="N", help="start epoch")
    parser.add_argument("--eval-every", default=1, type=int, metavar="N", help="eval the model every n epoch")
    parser.add_argument("--test-only", dest="test_only", help="Only test the model", action="store_true")
    parser.add_argument("--save-imgs", dest="save_imgs", action="store_true", help="save the predicted mask")

    parser.add_argument("--save-local", dest="save_local", help="save logs to local", action="store_true")

    # Distributed training parameters
    parser.add_argument("--world-size", default=1, type=int, help="number of distributed processes")
    parser.add_argument("--dist-url", default="env://", help="url used to set up distributed training")

    return parser


if __name__ == "__main__":
    args = get_args_parser().parse_args()

    output_root = "output"
    save_path = "{}_{}_{}/{date:%Y-%m-%d_%H:%M:%S}".format(
        args.model, args.train_dataset, args.data_cv, date=datetime.datetime.now()
    )
    args.output_dir = os.path.join(output_root, save_path)

    main(args)
