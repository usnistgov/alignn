"""Module for training script."""

from torch.nn.parallel import DistributedDataParallel as DDP
from functools import partial
from typing import Any, Dict, Union
import torch
import random
from sklearn.metrics import mean_absolute_error
import pickle as pk
import numpy as np
from torch import nn
from alignn.data import get_train_val_loaders
from alignn.config import TrainingConfig
from alignn.models.alignn_atomwise import ALIGNNAtomWise
from alignn.models.alignn_ff2 import ALIGNNFF2
from alignn.models.alignn_eff import ALIGNNeFF
from alignn.models.alignn import ALIGNN
from jarvis.db.jsonutils import dumpjson
import json
import pprint
import os
import warnings
import time
from sklearn.metrics import roc_auc_score
from alignn.utils import (
    # activated_output_transform,
    # make_standard_scalar_and_pca,
    # thresholded_output_transform,
    group_decay,
    setup_optimizer,
    print_train_val_loss,
)

# from sklearn.metrics import log_loss

warnings.filterwarnings("ignore", category=RuntimeWarning)

torch.autograd.detect_anomaly()


def train_dgl(
    config: Union[TrainingConfig, Dict[str, Any]],
    model: nn.Module = None,
    # checkpoint_dir: Path = Path("./"),
    train_val_test_loaders=[],
    rank=0,
    world_size=0,
    # log_tensorboard: bool = False,
):
    """Training entry point for DGL networks.

    `config` should conform to alignn.conf.TrainingConfig, and
    if passed as a dict with matching keys, pydantic validation is used
    """
    # print("rank", rank)
    # setup(rank, world_size)
    if rank == 0:
        print("config:")
        # print(config)
        if type(config) is dict:
            try:
                print(config)
                config = TrainingConfig(**config)
            except Exception as exp:
                print("Check", exp)

    if not os.path.exists(config.output_dir):
        os.makedirs(config.output_dir)
    # checkpoint_dir = os.path.join(config.output_dir)
    # deterministic = False
    classification = False
    tmp = config.dict()
    f = open(os.path.join(config.output_dir, "config.json"), "w")
    f.write(json.dumps(tmp, indent=4))
    f.close()
    global tmp_output_dir
    tmp_output_dir = config.output_dir
    pprint.pprint(tmp)  # , sort_dicts=False)
    if config.classification_threshold is not None:
        classification = True
    TORCH_DTYPES = {
        "float16": torch.float16,
        "float32": torch.float32,
        "float64": torch.float64,
        "bfloat": torch.bfloat16,
    }
    torch.set_default_dtype(TORCH_DTYPES[config.dtype])
    line_graph = False
    if config.model.alignn_layers > 0:
        line_graph = True
    if world_size > 1:
        use_ddp = True
    else:
        use_ddp = False
        device = "cpu"
        if torch.cuda.is_available():
            device = torch.device("cuda")
    if not train_val_test_loaders:
        # use input standardization for all real-valued feature sets
        # print("config.neighbor_strategy",config.neighbor_strategy)
        # import sys
        # sys.exit()
        (
            train_loader,
            val_loader,
            test_loader,
            prepare_batch,
        ) = get_train_val_loaders(
            dataset=config.dataset,
            target=config.target,
            n_train=config.n_train,
            n_val=config.n_val,
            n_test=config.n_test,
            train_ratio=config.train_ratio,
            val_ratio=config.val_ratio,
            test_ratio=config.test_ratio,
            batch_size=config.batch_size,
            atom_features=config.atom_features,
            neighbor_strategy=config.neighbor_strategy,
            standardize=config.atom_features != "cgcnn",
            line_graph=line_graph,
            id_tag=config.id_tag,
            pin_memory=config.pin_memory,
            workers=config.num_workers,
            save_dataloader=config.save_dataloader,
            use_canonize=config.use_canonize,
            filename=config.filename,
            cutoff=config.cutoff,
            max_neighbors=config.max_neighbors,
            output_features=config.model.output_features,
            classification_threshold=config.classification_threshold,
            target_multiplication_factor=config.target_multiplication_factor,
            standard_scalar_and_pca=config.standard_scalar_and_pca,
            keep_data_order=config.keep_data_order,
            output_dir=config.output_dir,
            use_lmdb=config.use_lmdb,
            dtype=config.dtype,
        )
    else:
        train_loader = train_val_test_loaders[0]
        val_loader = train_val_test_loaders[1]
        test_loader = train_val_test_loaders[2]
        prepare_batch = train_val_test_loaders[3]
    # rank=0
    if use_ddp:
        device = torch.device(f"cuda:{rank}")
    prepare_batch = partial(prepare_batch, device=device)
    if classification:
        config.model.classification = True
    _model = {
        "alignn_atomwise": ALIGNNAtomWise,
        "alignn": ALIGNN,
        "alignn_ff2": ALIGNNFF2,
        "alignn_eff": ALIGNNeFF,
    }
    if config.random_seed is not None:
        random.seed(config.random_seed)
        torch.manual_seed(config.random_seed)
        np.random.seed(config.random_seed)
        torch.cuda.manual_seed_all(config.random_seed)
        try:
            import torch_xla.core.xla_model as xm

            xm.set_rng_state(config.random_seed)
        except ImportError:
            pass
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        os.environ["PYTHONHASHSEED"] = str(config.random_seed)
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = str(":4096:8")
        torch.use_deterministic_algorithms(True)
    if model is None:
        net = _model.get(config.model.name)(config.model)
    else:
        net = model
    print("net parameters", sum(p.numel() for p in net.parameters()))
    # print("device", device)
    net.to(device)
    if use_ddp:
        net = DDP(net, device_ids=[rank], find_unused_parameters=True)
    # group parameters to skip weight decay for bias and batchnorm
    params = group_decay(net)
    optimizer = setup_optimizer(params, config)
    if config.scheduler == "none":
        # always return multiplier of 1 (i.e. do nothing)
        scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer, lambda epoch: 1.0
        )

    elif config.scheduler == "onecycle":
        steps_per_epoch = len(train_loader)
        # pct_start = config.warmup_steps / (config.epochs * steps_per_epoch)
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=config.learning_rate,
            epochs=config.epochs,
            steps_per_epoch=steps_per_epoch,
            # pct_start=pct_start,
            pct_start=0.3,
        )
    elif config.scheduler == "step":
        # pct_start = config.warmup_steps / (config.epochs * steps_per_epoch)
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer,
        )

    # if (
    #    config.model.name == "alignn_atomwise"
    #    or config.model.name == "alignn_ff2"
    # ):
    if "alignn_" in config.model.name:
        best_loss = np.inf
        criterion = nn.L1Loss()
        if classification:
            criterion = nn.NLLLoss()
        params = group_decay(net)
        optimizer = setup_optimizer(params, config)
        # optimizer = torch.optim.Adam(net.parameters(), lr=0.001)
        history_train = []
        history_val = []
        for e in range(config.epochs):
            # optimizer.zero_grad()
            train_init_time = time.time()
            running_loss = 0
            running_loss1 = 0
            running_loss2 = 0
            running_loss3 = 0
            running_loss4 = 0
            train_result = []
            for dats, jid in zip(train_loader, train_loader.dataset.ids):
                info = {}
                # info["id"] = jid
                optimizer.zero_grad()
                if (config.model.alignn_layers) > 0:
                    result = net(
                        [
                            dats[0].to(device),
                            dats[1].to(device),
                            dats[2].to(device),
                        ]
                    )
                    # result = net(dats[0].to(device), dats[2].to(device),dats[1].to(device))
                    # result = net([dats[0].to(device), dats[1].to(device),lat=dats[2].to(device)])
                    # batched_graph, batched_line_graph, torch.stack(lattices),torch.tensor(labels)

                else:
                    result = net(dats[0].to(device))
                # info = {}
                info["target_out"] = []
                info["pred_out"] = []
                info["target_atomwise_pred"] = []
                info["pred_atomwise_pred"] = []
                info["target_grad"] = []
                info["pred_grad"] = []
                info["target_stress"] = []
                info["pred_stress"] = []

                loss1 = 0  # Such as energy
                loss2 = 0  # Such as bader charges
                loss3 = 0  # Such as forces
                loss4 = 0  # Such as stresses
                if config.model.output_features is not None:
                    loss1 = config.model.graphwise_weight * criterion(
                        result["out"],
                        dats[-1].to(device),
                        # result["out"], dats[2].to(device)
                    )
                    info["target_out"] = dats[-1].cpu().numpy().tolist()
                    # info["target_out"] = dats[2].cpu().numpy().tolist()
                    info["pred_out"] = (
                        result["out"].cpu().detach().numpy().tolist()
                    )
                    running_loss1 += loss1.item()
                if (
                    config.model.atomwise_output_features > 0
                    # config.model.atomwise_output_features is not None
                    and config.model.atomwise_weight != 0
                ):
                    loss2 = config.model.atomwise_weight * criterion(
                        result["atomwise_pred"].to(device),
                        dats[0].ndata["atomwise_target"].to(device),
                    )
                    info["target_atomwise_pred"] = (
                        dats[0].ndata["atomwise_target"].cpu().numpy().tolist()
                    )
                    info["pred_atomwise_pred"] = (
                        result["atomwise_pred"].cpu().detach().numpy().tolist()
                    )
                    running_loss2 += loss2.item()

                if config.model.calculate_gradient:
                    loss3 = config.model.gradwise_weight * criterion(
                        result["grad"].to(device),
                        dats[0].ndata["atomwise_grad"].to(device),
                    )
                    info["target_grad"] = (
                        dats[0].ndata["atomwise_grad"].cpu().numpy().tolist()
                    )
                    info["pred_grad"] = (
                        result["grad"].cpu().detach().numpy().tolist()
                    )
                    running_loss3 += loss3.item()
                if config.model.stresswise_weight != 0:
                    loss4 = config.model.stresswise_weight * criterion(
                        (result["stresses"]).to(device),
                        torch.cat(tuple(dats[0].ndata["stresses"])).to(device),
                    )
                    info["target_stress"] = (
                        torch.cat(tuple(dats[0].ndata["stresses"]))
                        .cpu()
                        .numpy()
                        .tolist()
                        # dats[0].ndata["stresses"][0].cpu().numpy().tolist()
                    )
                    info["pred_stress"] = (
                        result["stresses"].cpu().detach().numpy().tolist()
                    )
                    running_loss4 += loss4.item()
                    # print("target_stress", info["target_stress"][0])
                    # print("pred_stress", info["pred_stress"][0])
                train_result.append(info)
                loss = loss1 + loss2 + loss3 + loss4
                loss.backward()
                optimizer.step()
                # optimizer.zero_grad() #never
                running_loss += loss.item()
            # mean_out, mean_atom, mean_grad, mean_stress = get_batch_errors(
            #    train_result
            # )
            # dumpjson(filename="Train_results.json", data=train_result)
            scheduler.step()
            train_final_time = time.time()
            train_ep_time = train_final_time - train_init_time
            # if rank == 0: # or world_size == 1:
            history_train.append(
                [
                    running_loss,
                    running_loss1,
                    running_loss2,
                    running_loss3,
                    running_loss4,
                ]
            )
            dumpjson(
                filename=os.path.join(config.output_dir, "history_train.json"),
                data=history_train,
            )
            val_loss = 0
            val_loss1 = 0
            val_loss2 = 0
            val_loss3 = 0
            val_loss4 = 0
            val_result = []
            # for dats in val_loader:
            val_init_time = time.time()
            for dats, jid in zip(val_loader, val_loader.dataset.ids):
                info = {}
                info["id"] = jid
                optimizer.zero_grad()
                # result = net([dats[0].to(device), dats[1].to(device)])
                if (config.model.alignn_layers) > 0:
                    # result = net([dats[0].to(device), dats[2].to(device),  dats[1].to(device)])
                    # result = net(dats[0].to(device), dats[2].to(device),dats[1].to(device))
                    result = net(
                        [
                            dats[0].to(device),
                            dats[1].to(device),
                            dats[2].to(device),
                        ]
                    )
                else:
                    result = net(dats[0].to(device))
                # info = {}
                info["target_out"] = []
                info["pred_out"] = []
                info["target_atomwise_pred"] = []
                info["pred_atomwise_pred"] = []
                info["target_grad"] = []
                info["pred_grad"] = []
                info["target_stress"] = []
                info["pred_stress"] = []
                loss1 = 0  # Such as energy
                loss2 = 0  # Such as bader charges
                loss3 = 0  # Such as forces
                loss4 = 0  # Such as stresses
                if config.model.output_features is not None:
                    loss1 = config.model.graphwise_weight * criterion(
                        result["out"], dats[-1].to(device)
                    )
                    info["target_out"] = dats[-1].cpu().numpy().tolist()
                    info["pred_out"] = (
                        result["out"].cpu().detach().numpy().tolist()
                    )
                    val_loss1 += loss1.item()

                if (
                    config.model.atomwise_output_features > 0
                    and config.model.atomwise_weight != 0
                ):
                    loss2 = config.model.atomwise_weight * criterion(
                        result["atomwise_pred"].to(device),
                        dats[0].ndata["atomwise_target"].to(device),
                    )
                    info["target_atomwise_pred"] = (
                        dats[0].ndata["atomwise_target"].cpu().numpy().tolist()
                    )
                    info["pred_atomwise_pred"] = (
                        result["atomwise_pred"].cpu().detach().numpy().tolist()
                    )
                    val_loss2 += loss2.item()
                if config.model.calculate_gradient:
                    loss3 = config.model.gradwise_weight * criterion(
                        result["grad"].to(device),
                        dats[0].ndata["atomwise_grad"].to(device),
                    )
                    info["target_grad"] = (
                        dats[0].ndata["atomwise_grad"].cpu().numpy().tolist()
                    )
                    info["pred_grad"] = (
                        result["grad"].cpu().detach().numpy().tolist()
                    )
                    val_loss3 += loss3.item()
                if config.model.stresswise_weight != 0:
                    # loss4 = config.model.stresswise_weight * criterion(
                    #    result["stress"].to(device),
                    #    dats[0].ndata["stresses"][0].to(device),
                    # )
                    loss4 = config.model.stresswise_weight * criterion(
                        (result["stresses"]).to(device),
                        torch.cat(tuple(dats[0].ndata["stresses"])).to(device),
                    )
                    info["target_stress"] = (
                        torch.cat(tuple(dats[0].ndata["stresses"]))
                        .cpu()
                        .numpy()
                        .tolist()
                        # dats[0].ndata["stresses"][0].cpu().numpy().tolist()
                    )
                    info["pred_stress"] = (
                        result["stresses"].cpu().detach().numpy().tolist()
                    )
                    val_loss4 += loss4.item()
                loss = loss1 + loss2 + loss3 + loss4
                val_result.append(info)
                val_loss += loss.item()
            # mean_out, mean_atom, mean_grad, mean_stress = get_batch_errors(
            #    val_result
            # )
            val_fin_time = time.time()
            val_ep_time = val_fin_time - val_init_time
            current_model_name = "current_model.pt"
            torch.save(
                net.state_dict(),
                os.path.join(config.output_dir, current_model_name),
            )
            saving_msg = ""
            if val_loss < best_loss:
                best_loss = val_loss
                best_model_name = "best_model.pt"
                torch.save(
                    net.state_dict(),
                    os.path.join(config.output_dir, best_model_name),
                )
                # print("Saving data for epoch:", e)
                saving_msg = "Saving model"
                dumpjson(
                    filename=os.path.join(
                        config.output_dir, "Train_results.json"
                    ),
                    data=train_result,
                )
                dumpjson(
                    filename=os.path.join(
                        config.output_dir, "Val_results.json"
                    ),
                    data=val_result,
                )
                best_model = net
            history_train.append(
                [val_loss, val_loss1, val_loss2, val_loss3, val_loss4]
            )
            # history_val.append([mean_out, mean_atom, mean_grad, mean_stress])
            dumpjson(
                filename=os.path.join(config.output_dir, "history_val.json"),
                data=history_val,
            )
            if rank == 0:
                print_train_val_loss(
                    e,
                    running_loss,
                    running_loss1,
                    running_loss2,
                    running_loss3,
                    running_loss4,
                    val_loss,
                    val_loss1,
                    val_loss2,
                    val_loss3,
                    val_loss4,
                    train_ep_time,
                    val_ep_time,
                    saving_msg=saving_msg,
                )

        if rank == 0 or world_size == 1:
            test_loss = 0
            test_result = []
            for dats, jid in zip(test_loader, test_loader.dataset.ids):
                # for dats in test_loader:
                info = {}
                info["id"] = jid
                optimizer.zero_grad()
                if (config.model.alignn_layers) > 0:
                    # result = net([dats[0].to(device), dats[1].to(device)])
                    result = net(
                        [
                            dats[0].to(device),
                            dats[1].to(device),
                            dats[2].to(device),
                        ]
                    )
                else:
                    result = net(dats[0].to(device))
                loss1 = 0  # Such as energy
                loss2 = 0  # Such as bader charges
                loss3 = 0  # Such as forces
                loss4 = 0  # Such as stresses
                if (
                    config.model.output_features is not None
                    and not classification
                ):
                    # print('result["out"]',result["out"])
                    # print('dats[2]',dats[2])
                    loss1 = config.model.graphwise_weight * criterion(
                        result["out"], dats[-1].to(device)
                    )
                    info["target_out"] = dats[-1].cpu().numpy().tolist()
                    info["pred_out"] = (
                        result["out"].cpu().detach().numpy().tolist()
                    )

                if config.model.atomwise_output_features > 0:
                    loss2 = config.model.atomwise_weight * criterion(
                        result["atomwise_pred"].to(device),
                        dats[0].ndata["atomwise_target"].to(device),
                    )
                    info["target_atomwise_pred"] = (
                        dats[0].ndata["atomwise_target"].cpu().numpy().tolist()
                    )
                    info["pred_atomwise_pred"] = (
                        result["atomwise_pred"].cpu().detach().numpy().tolist()
                    )

                if config.model.calculate_gradient:
                    loss3 = config.model.gradwise_weight * criterion(
                        result["grad"].to(device),
                        dats[0].ndata["atomwise_grad"].to(device),
                    )
                    info["target_grad"] = (
                        dats[0].ndata["atomwise_grad"].cpu().numpy().tolist()
                    )
                    info["pred_grad"] = (
                        result["grad"].cpu().detach().numpy().tolist()
                    )
                if config.model.stresswise_weight != 0:
                    loss4 = config.model.stresswise_weight * criterion(
                        result["stresses"].to(device),
                        torch.cat(tuple(dats[0].ndata["stresses"])).to(device),
                    )
                    info["target_stress"] = (
                        torch.cat(tuple(dats[0].ndata["stresses"]))
                        .cpu()
                        .numpy()
                        .tolist()
                    )
                    info["pred_stress"] = (
                        result["stresses"].cpu().detach().numpy().tolist()
                    )
                test_result.append(info)
                loss = loss1 + loss2 + loss3 + loss4
                if not classification:
                    test_loss += loss.item()
            print("TestLoss", e, test_loss)
            dumpjson(
                filename=os.path.join(config.output_dir, "Test_results.json"),
                data=test_result,
            )
            last_model_name = "last_model.pt"
            torch.save(
                net.state_dict(),
                os.path.join(config.output_dir, last_model_name),
            )
            # return test_result
    if rank == 0 or world_size == 1:
        if config.write_predictions and classification:
            best_model.eval()
            # net.eval()
            f = open(
                os.path.join(
                    config.output_dir, "prediction_results_test_set.csv"
                ),
                "w",
            )
            f.write("id,target,prediction\n")
            targets = []
            predictions = []
            with torch.no_grad():
                ids = test_loader.dataset.ids  # [test_loader.dataset.indices]
                for dat, id in zip(test_loader, ids):
                    g, lg, target = dat
                    out_data = best_model([g.to(device), lg.to(device)])["out"]
                    # out_data = net([g.to(device), lg.to(device)])["out"]
                    # out_data = torch.exp(out_data.cpu())
                    # print('target',target)
                    # print('out_data',out_data)
                    top_p, top_class = torch.topk(torch.exp(out_data), k=1)
                    target = int(target.cpu().numpy().flatten().tolist()[0])

                    f.write("%s, %d, %d\n" % (id, (target), (top_class)))
                    targets.append(target)
                    predictions.append(
                        top_class.cpu().numpy().flatten().tolist()[0]
                    )
            f.close()

            print("predictions", predictions)
            print("targets", targets)
            print(
                "Test ROCAUC:",
                roc_auc_score(np.array(targets), np.array(predictions)),
            )

        if (
            config.write_predictions
            and not classification
            and config.model.output_features > 1
        ):
            best_model.eval()
            # net.eval()
            mem = []
            with torch.no_grad():
                ids = test_loader.dataset.ids  # [test_loader.dataset.indices]
                for dat, id in zip(test_loader, ids):
                    g, lg, target = dat
                    out_data = best_model([g.to(device), lg.to(device)])["out"]
                    # out_data = net([g.to(device), lg.to(device)])["out"]
                    out_data = out_data.cpu().numpy().tolist()
                    if config.standard_scalar_and_pca:
                        sc = pk.load(open("sc.pkl", "rb"))
                        out_data = list(
                            sc.transform(np.array(out_data).reshape(1, -1))[0]
                        )  # [0][0]
                    target = target.cpu().numpy().flatten().tolist()
                    info = {}
                    info["id"] = id
                    info["target"] = target
                    info["predictions"] = out_data
                    mem.append(info)
            dumpjson(
                filename=os.path.join(
                    config.output_dir, "multi_out_predictions.json"
                ),
                data=mem,
            )
        if (
            config.write_predictions
            and not classification
            and config.model.output_features == 1
            and config.model.gradwise_weight == 0
        ):
            best_model.eval()
            # net.eval()
            f = open(
                os.path.join(
                    config.output_dir, "prediction_results_test_set.csv"
                ),
                "w",
            )
            f.write("id,target,prediction\n")
            targets = []
            predictions = []
            with torch.no_grad():
                ids = test_loader.dataset.ids  # [test_loader.dataset.indices]
                for dat, id in zip(test_loader, ids):
                    g, lg, target = dat
                    out_data = best_model([g.to(device), lg.to(device)])["out"]
                    # out_data = net([g.to(device), lg.to(device)])["out"]
                    out_data = out_data.cpu().numpy().tolist()
                    if config.standard_scalar_and_pca:
                        sc = pk.load(
                            open(os.path.join(tmp_output_dir, "sc.pkl"), "rb")
                        )
                        out_data = sc.transform(
                            np.array(out_data).reshape(-1, 1)
                        )[0][0]
                    target = target.cpu().numpy().flatten().tolist()
                    if len(target) == 1:
                        target = target[0]
                    f.write("%s, %6f, %6f\n" % (id, target, out_data))
                    targets.append(target)
                    predictions.append(out_data)
            f.close()

            print(
                "Test MAE:",
                mean_absolute_error(np.array(targets), np.array(predictions)),
            )
            best_model.eval()
            # net.eval()
            f = open(
                os.path.join(
                    config.output_dir, "prediction_results_train_set.csv"
                ),
                "w",
            )
            f.write("target,prediction\n")
            targets = []
            predictions = []
            with torch.no_grad():
                ids = train_loader.dataset.ids  # [test_loader.dataset.indices]
                for dat, id in zip(train_loader, ids):
                    g, lg, target = dat
                    out_data = best_model([g.to(device), lg.to(device)])["out"]
                    # out_data = net([g.to(device), lg.to(device)])["out"]
                    out_data = out_data.cpu().numpy().tolist()
                    if config.standard_scalar_and_pca:
                        sc = pk.load(
                            open(os.path.join(tmp_output_dir, "sc.pkl"), "rb")
                        )
                        out_data = sc.transform(
                            np.array(out_data).reshape(-1, 1)
                        )[0][0]
                    target = target.cpu().numpy().flatten().tolist()
                    # if len(target) == 1:
                    #    target = target[0]
                    # if len(out_data) == 1:
                    #    out_data = out_data[0]
                    for ii, jj in zip(target, out_data):
                        f.write("%6f, %6f\n" % (ii, jj))
                        targets.append(ii)
                        predictions.append(jj)
            f.close()
        if config.use_lmdb:
            print("Closing LMDB.")
            train_loader.dataset.close()
            val_loader.dataset.close()
            test_loader.dataset.close()


if __name__ == "__main__":
    config = TrainingConfig(
        random_seed=123, epochs=10, n_train=32, n_val=32, batch_size=16
    )
    history = train_dgl(config)
