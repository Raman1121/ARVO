import os

os.environ["TORCH_HOME"] = os.path.dirname(os.getcwd())

import datetime
import random
import re
import time
import json
import warnings
import timm
import pandas as pd
import numpy as np
from utilities import presets
import torch
import torch.utils.data
import torchvision
import utilities.transforms
from utilities.utils import *
from utilities.training_utils import *
from parse_args import *
from utilities.sampler import RASampler
from torch import nn
from torch.utils.data.dataloader import default_collate
from torchvision.transforms.functional import InterpolationMode
import yaml
from pprint import pprint


def create_results_df(args):
    test_results_df = None

    if args.sens_attribute == "gender":
        test_results_df = pd.DataFrame(
            columns=[
                "Tuning Method",
                "Train Percent",
                "LR",
                "Test AUC Overall",
                "Test AUC Male",
                "Test AUC Female",
                "Test AUC Difference",
                "EquiOdd_diff",
                "EquiOdd_ratio",
                "DPD",
                "DPR",
            ]
        )

    elif (
        args.sens_attribute == "skin_type"
        or args.sens_attribute == "age"
        or args.sens_attribute == "race"
        or args.sens_attribute == "age_sex"
    ):
        cols = [
            "Tuning Method",
            "Train Percent",
            "LR",
            "Test AUC Overall",
            "Test AUC (Best)",
            "Test AUC (Worst)",
            "Test AUC Difference",
            "EquiOdd_diff",
            "EquiOdd_ratio",
            "DPD",
            "DPR",
        ]

        test_results_df = pd.DataFrame(columns=cols)
    else:
        raise NotImplementedError("Sensitive attribute not implemented")

    return test_results_df


def main(args):
    assert args.sens_attribute is not None, "Sensitive attribute not provided"

    os.makedirs(args.fig_savepath, exist_ok=True)

    # Making directory for saving checkpoints
    if args.output_dir:
        utils.mkdir(args.output_dir)
        utils.mkdir(os.path.join(args.output_dir, "checkpoints"))

    try:
        test_results_df = pd.read_csv(
            os.path.join(args.output_dir, args.test_results_df)
        )
        print("Reading existing results dataframe")
    except:
        print("Creating new results dataframe")
        test_results_df = create_results_df(args)

    utils.init_distributed_mode(args)
    print(args)

    device = torch.device(args.device)

    if args.use_deterministic_algorithms:
        torch.backends.cudnn.benchmark = False
        torch.use_deterministic_algorithms(True)
    else:
        torch.backends.cudnn.benchmark = True

    ## CREATING DATASETS AND DATALOADERS

    with open("config.yaml") as file:
        yaml_data = yaml.safe_load(file)

    (
        dataset,
        dataset_val,
        dataset_test,
        train_sampler,
        val_sampler,
        test_sampler,
    ) = get_fairness_data(args, yaml_data)

    args.num_classes = len(dataset.classes)
    print("DATASET: ", args.dataset)
    print("Size of training dataset: ", len(dataset))
    print("Size of validation dataset: ", len(dataset_val))
    print("Size of test dataset: ", len(dataset_test))
    print("Number of classes: ", args.num_classes)

    collate_fn = None
    mixup_transforms = get_mixup_transforms(args)

    if mixup_transforms:
        mixupcutmix = torchvision.transforms.RandomChoice(mixup_transforms)

        def collate_fn(batch):
            return mixupcutmix(*default_collate(batch))

    if args.dataset == "papila":
        drop_last = False
    else:
        drop_last = True

    data_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        sampler=train_sampler,
        num_workers=args.workers,
        pin_memory=True,
        collate_fn=collate_fn,
        drop_last=drop_last,
    )
    data_loader_val = torch.utils.data.DataLoader(
        dataset_val,
        batch_size=args.batch_size,
        sampler=val_sampler,
        num_workers=args.workers,
        pin_memory=True,
        # drop_last=True
    )
    data_loader_test = torch.utils.data.DataLoader(
        dataset_test,
        batch_size=args.batch_size,
        sampler=test_sampler,
        num_workers=args.workers,
        pin_memory=True,
        drop_last=False,
    )

    MODEL_NAMES = {
        "resnet50": "resnet50",
        "densenet121": "densenet121",
        "nasnet": "nasnetalarge",
        "enet_b0": "efficientnet_b0",
        "enet_b3": "efficientnet_b3",
        "vit_base": "vit_base_patch16_224",
        "vit_base_miil": "vit_base_patch16_224_miil",
        "vit_base_clip": "vit_base_patch16_clip_224",
        "vit_base_mae": "vit_base_patch16_224.mae",
        "vit_base_dino": "vit_base_patch16_224.dino",
    }

    # CREATING THE MODEL
    print("TUNING METHOD: ", args.tuning_method)
    print("Creating model")

    model = timm.create_model(
        MODEL_NAMES[args.model], pretrained=True, num_classes=args.num_classes
    )

    # Check Tunable Params
    trainable_params, all_param = utils.check_tunable_params(model, False)
    trainable_percentage = 100 * trainable_params / all_param

    model.to(device)

    # Computing class weights for a weighted loss in case of highly imbalanced datasets
    if args.compute_cw:
        if args.dataset == "papila":
            weight = torch.tensor([0.20714285714285716, 0.7928571428571428])
        weight = weight.to(device)
        print("Using CW Loss with weights: ", weight)
    else:
        weight = None

    criterion = nn.CrossEntropyLoss(
        label_smoothing=args.label_smoothing, reduction="none", weight=weight
    )

    ece_criterion = utils.ECELoss()

    custom_keys_weight_decay = []
    if args.bias_weight_decay is not None:
        custom_keys_weight_decay.append(("bias", args.bias_weight_decay))
    if args.transformer_embedding_decay is not None:
        for key in [
            "class_token",
            "position_embedding",
            "relative_position_bias_table",
        ]:
            custom_keys_weight_decay.append((key, args.transformer_embedding_decay))

    parameters = utils.set_weight_decay(
        model,
        args.weight_decay,
        norm_weight_decay=args.norm_weight_decay,
        custom_keys_weight_decay=custom_keys_weight_decay
        if len(custom_keys_weight_decay) > 0
        else None,
    )

    # Optimizer
    optimizer = get_optimizer(args, parameters)
    scaler = torch.cuda.amp.GradScaler() if args.amp else None

    # LR Scheduler
    lr_scheduler = get_lr_scheduler(args, optimizer)

    if args.disable_training:
        print("Training Process Skipped")
    else:
        print("Start training")
        model_ema = None
        start_time = time.time()
        for epoch in range(args.start_epoch, args.epochs):
            if args.distributed:
                train_sampler.set_epoch(epoch)
            (
                train_acc,
                train_best_acc,
                train_worst_acc,
                train_auc,
                train_best_auc,
                train_worst_auc,
            ) = train_one_epoch_fairness(
                model,
                criterion,
                ece_criterion,
                optimizer,
                data_loader,
                device,
                epoch,
                args,
                model_ema,
                scaler,
            )
            lr_scheduler.step()

            if args.sens_attribute == "gender":
                if args.cal_equiodds:
                    (
                        val_acc,
                        val_male_acc,
                        val_female_acc,
                        val_auc,
                        val_male_auc,
                        val_female_auc,
                        val_loss,
                        val_max_loss,
                        equiodds_diff,
                        equiodds_ratio,
                        dpd,
                        dpr,
                    ) = evaluate_fairness_gender(
                        model,
                        criterion,
                        ece_criterion,
                        data_loader_val,
                        args=args,
                        device=device,
                    )
                else:
                    (
                        val_acc,
                        val_male_acc,
                        val_female_acc,
                        val_auc,
                        val_male_auc,
                        val_female_auc,
                        val_loss,
                        val_max_loss,
                    ) = evaluate_fairness_gender(
                        model,
                        criterion,
                        ece_criterion,
                        data_loader_val,
                        args=args,
                        device=device,
                    )

                best_val_acc = max(val_male_acc, val_female_acc)
                worst_val_acc = min(val_male_acc, val_female_acc)

                best_val_auc = max(val_male_auc, val_female_auc)
                worst_val_auc = min(val_male_auc, val_female_auc)

                print(
                    "Val Acc: {:.2f}, Val Male Acc {:.2f}, Val Female Acc {:.2f}, Val Loss: {:.2f}, Val MAX LOSS: {:.2f}".format(
                        val_acc,
                        val_male_acc,
                        val_female_acc,
                        torch.mean(val_loss),
                        val_max_loss,
                    )
                )
                print(
                    "Val AUC: {:.2f}, Val Male AUC {:.2f}, Val Female AUC {:.2f}".format(
                        val_auc,
                        val_male_auc,
                        val_female_auc,
                    )
                )

            elif args.sens_attribute == "skin_type":
                assert args.skin_type == "binary"
                assert args.cal_equiodds is not None

                (
                    val_acc,
                    val_acc_type0,
                    val_acc_type1,
                    val_auc,
                    val_auc_type0,
                    val_auc_type1,
                    val_loss,
                    val_max_loss,
                    equiodds_diff,
                    equiodds_ratio,
                    dpd,
                    dpr,
                ) = evaluate_fairness_skin_type_binary(
                    model,
                    criterion,
                    ece_criterion,
                    data_loader_val,
                    args=args,
                    device=device,
                )

                best_val_acc = max(val_acc_type0, val_acc_type1)
                worst_val_acc = min(val_acc_type0, val_acc_type1)

                best_val_auc = max(val_auc_type0, val_auc_type1)
                worst_val_auc = min(val_auc_type0, val_auc_type1)

                print(
                    "Val Acc: {:.2f}, Val Type 0 Acc: {:.2f}, Val Type 1 Acc: {:.2f}, Val Loss: {:.2f}, Val MAX LOSS: {:.2f}".format(
                        val_acc,
                        val_acc_type0,
                        val_acc_type1,
                        torch.mean(val_loss),
                        val_max_loss,
                    )
                )
                print(
                    "Val AUC: {:.2f}, Val Type 0 AUC: {:.2f}, Val Type 1 AUC: {:.2f}".format(
                        val_auc,
                        val_auc_type0,
                        val_auc_type1,
                    )
                )
                print("\n")

            elif args.sens_attribute == "age":
                assert args.age_type == "binary"
                assert args.cal_equiodds is not None

                (
                    val_acc,
                    acc_age0_avg,
                    acc_age1_avg,
                    val_auc,
                    auc_age0_avg,
                    auc_age1_avg,
                    val_loss,
                    val_max_loss,
                    equiodds_diff,
                    equiodds_ratio,
                    dpd,
                    dpr,
                ) = evaluate_fairness_age_binary(
                    model,
                    criterion,
                    ece_criterion,
                    data_loader_val,
                    args=args,
                    device=device,
                )

                best_val_acc = max(acc_age0_avg, acc_age1_avg)
                worst_val_acc = min(acc_age0_avg, acc_age1_avg)

                best_val_auc = max(auc_age0_avg, auc_age1_avg)
                worst_val_auc = min(auc_age0_avg, auc_age1_avg)

                print(
                    "Val Acc: {:.2f}, Val Age Group0 Acc: {:.2f}, Val Age Group1 Acc: {:.2f}, Val Loss: {:.2f}, Val MAX LOSS: {:.2f}".format(
                        val_acc,
                        acc_age0_avg,
                        acc_age1_avg,
                        torch.mean(val_loss),
                        val_max_loss,
                    )
                )
                print(
                    "Val AUC: {:.2f}, Val Age Group0 AUC: {:.2f}, Val Age Group1 AUC: {:.2f}".format(
                        val_auc,
                        auc_age0_avg,
                        auc_age1_avg,
                    )
                )
                print("\n")

            elif args.sens_attribute == "race":
                assert args.cal_equiodds is not None
                (
                    val_acc,
                    acc_race0_avg,
                    acc_race1_avg,
                    val_auc,
                    auc_race0_avg,
                    auc_race1_avg,
                    val_loss,
                    val_max_loss,
                    equiodds_diff,
                    equiodds_ratio,
                    dpd,
                    dpr,
                ) = evaluate_fairness_race_binary(
                    model,
                    criterion,
                    ece_criterion,
                    data_loader_val,
                    args=args,
                    device=device,
                )

            else:
                raise NotImplementedError("Sensitive attribute not implemented")

        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        print("Training time {}".format(total_time_str))

        # Obtaining the performance on test set
        print("Obtaining the performance on test set")

        if args.sens_attribute == "gender":
            assert args.cal_equiodds is not None
            (
                test_acc,
                test_male_acc,
                test_female_acc,
                test_auc,
                test_male_auc,
                test_female_auc,
                test_loss,
                test_max_loss,
                equiodds_diff,
                equiodds_ratio,
                dpd,
                dpr,
            ) = evaluate_fairness_gender(
                model,
                criterion,
                ece_criterion,
                data_loader_test,
                args=args,
                device=device,
            )

            print("\n")
            print("Overall Test Accuracy: ", test_acc)
            print("Test Male Accuracy: ", test_male_acc)
            print("Test Female Accuracy: ", test_female_acc)
            print("\n")
            print("Overall Test AUC: ", test_auc)
            print("Test Male AUC: ", test_male_auc)
            print("Test Female AUC: ", test_female_auc)
            if args.cal_equiodds:
                print("\n")
                print("EquiOdds Difference: ", equiodds_diff)
                print("EquiOdds Ratio: ", equiodds_ratio)
                print("DPD: ", dpd)
                print("DPR: ", dpr)

        elif args.sens_attribute == "skin_type":
            assert args.skin_type == "binary"
            assert args.cal_equiodds is not None
            (
                test_acc,
                test_acc_type0,
                test_acc_type1,
                test_auc,
                test_auc_type0,
                test_auc_type1,
                test_loss,
                test_max_loss,
                equiodds_diff,
                equiodds_ratio,
                dpd,
                dpr,
            ) = evaluate_fairness_skin_type_binary(
                model,
                criterion,
                ece_criterion,
                data_loader_test,
                args=args,
                device=device,
            )

            print("\n")
            print("Overall Test accuracy: ", test_acc)
            print("Test Type 0 Accuracy: ", test_acc_type0)
            print("Test Type 1 Accuracy: ", test_acc_type1)
            print("\n")
            print("Overall Test AUC: ", test_auc)
            print("Test Type 0 AUC: ", test_auc_type0)
            print("Test Type 1 AUC: ", test_auc_type1)
            if args.cal_equiodds:
                print("\n")
                print("EquiOdds Difference: ", equiodds_diff)
                print("EquiOdds Ratio: ", equiodds_ratio)
                print("DPD: ", dpd)
                print("DPR: ", dpr)

        elif args.sens_attribute == "age":
            assert args.age_type == "binary"
            assert args.cal_equiodds is not None

            (
                test_acc,
                test_acc_type0,
                test_acc_type1,
                test_auc,
                test_auc_type0,
                test_auc_type1,
                test_loss,
                test_max_loss,
                equiodds_diff,
                equiodds_ratio,
                dpd,
                dpr,
            ) = evaluate_fairness_age_binary(
                model,
                criterion,
                ece_criterion,
                data_loader_test,
                args=args,
                device=device,
            )

            print("\n")
            print("Overall Test accuracy: ", test_acc)
            print("Test Age Group 0 Accuracy: ", test_acc_type0)
            print("Test Age Group 1 Accuracy: ", test_acc_type1)
            print("\n")
            print("Overall Test AUC: ", test_auc)
            print("Test Age Group 0 AUC: ", test_auc_type0)
            print("Test Age Group 1 AUC: ", test_auc_type1)
            if args.cal_equiodds:
                print("\n")
                print("EquiOdds Difference: ", equiodds_diff)
                print("EquiOdds Ratio: ", equiodds_ratio)
                print("DPD: ", dpd)
                print("DPR: ", dpr)

        elif args.sens_attribute == "race":
            (
                test_acc,
                test_acc_type0,
                test_acc_type1,
                test_auc,
                test_auc_type0,
                test_auc_type1,
                test_loss,
                test_max_loss,
                equiodds_diff,
                equiodds_ratio,
                dpd,
                dpr,
            ) = evaluate_fairness_race_binary(
                model,
                criterion,
                ece_criterion,
                data_loader_test,
                args=args,
                device=device,
            )

            print("\n")
            print("Overall Test accuracy: ", test_acc)
            print("Test Race Group 0 Accuracy: ", test_acc_type0)
            print("Test Race Group 1 Accuracy: ", test_acc_type1)
            print("\n")
            print("Overall Test AUC: ", test_auc)
            print("Test Race Group 0 AUC: ", test_auc_type0)
            print("Test Race Group 1 AUC: ", test_auc_type1)
            if args.cal_equiodds:
                print("\n")
                print("EquiOdds Difference: ", equiodds_diff)
                print("EquiOdds Ratio: ", equiodds_ratio)
                print("DPD: ", dpd)
                print("DPR: ", dpr)

        else:
            raise NotImplementedError("Sensitive Attribute not supported")

        print("Test loss: ", round(torch.mean(test_loss).item(), 3))
        print("Test max loss: ", round(test_max_loss.item(), 3))

        # Add these results to CSV

        if args.sens_attribute == "gender":
            assert args.use_metric == "auc"

            _new_row = [
                args.tuning_method,
                round(trainable_percentage, 3),
                args.lr,
                test_auc,
                test_male_auc,
                test_female_auc,
                round(abs(test_male_auc - test_female_auc), 3),
                round(equiodds_diff, 3),
                round(equiodds_ratio, 3),
                round(dpd, 3),
                round(dpr, 3),
            ]

        elif args.sens_attribute == "skin_type":
            assert args.skin_type == "binary"
            assert args.use_metric == "auc"
            assert args.cal_equiodds is not None

            best_auc = max(test_auc_type0, test_auc_type1)
            worst_auc = min(test_auc_type0, test_auc_type1)

            _new_row = [
                args.tuning_method,
                round(trainable_percentage, 3),
                args.lr,
                test_auc,
                best_auc,
                worst_auc,
                round(abs(best_auc - worst_auc), 3),
                round(equiodds_diff, 3),
                round(equiodds_ratio, 3),
                round(dpd, 3),
                round(dpr, 3),
            ]

        elif args.sens_attribute == "age":
            assert args.age_type == "binary"
            assert args.use_metric == "auc"
            assert args.cal_equiodds is not None

            best_auc = max(test_auc_type0, test_auc_type1)
            worst_auc = min(test_auc_type0, test_auc_type1)

            _new_row = [
                args.tuning_method,
                round(trainable_percentage, 3),
                args.lr,
                test_auc,
                best_auc,
                worst_auc,
                round(abs(best_auc - worst_auc), 3),
                round(equiodds_diff, 3),
                round(equiodds_ratio, 3),
                round(dpd, 3),
                round(dpr, 3),
            ]

        elif args.sens_attribute == "race":
            assert args.use_metric == "auc"
            assert args.cal_equiodds is not None

            best_auc = max(test_auc_type0, test_auc_type1)
            worst_auc = min(test_auc_type0, test_auc_type1)

            _new_row = [
                args.tuning_method,
                round(trainable_percentage, 3),
                args.lr,
                test_auc,
                best_auc,
                worst_auc,
                round(abs(best_auc - worst_auc), 3),
                round(equiodds_diff, 3),
                round(equiodds_ratio, 3),
                round(dpd, 3),
                round(dpr, 3),
            ]

        else:
            raise NotImplementedError("Sensitive attribute not implemented")

        test_results_df.loc[len(test_results_df)] = _new_row

        print(
            "Saving test results df at: {}".format(
                os.path.join(args.output_dir, args.test_results_df)
            )
        )

        test_results_df.to_csv(
            os.path.join(args.output_dir, args.test_results_df), index=False
        )


if __name__ == "__main__":
    args = get_args_parser().parse_args()
    args.output_dir = os.path.join(os.getcwd(), 'results', args.model, args.dataset)

    if "auc" in args.objective_metric:
        args.use_metric = "auc"

    assert args.use_metric == "auc"
    assert args.cal_equiodds is not None

    args.test_results_df = (
        "RESULTS_" + args.sens_attribute + "_" + args.objective_metric + ".csv"
    )

    current_wd = os.getcwd()
    args.fig_savepath = os.path.join(args.output_dir, "plots/")

    args.train_fscl_classifier = False
    args.train_fscl_encoder = False

    main(args)
