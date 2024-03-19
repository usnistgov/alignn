"""Ignite training script.

from the repository root, run
`PYTHONPATH=$PYTHONPATH:. python alignn/train.py`
then `tensorboard --logdir tb_logs/test` to monitor results...
"""

from functools import partial
from typing import Any, Dict, Union
import torch
import random
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import log_loss
import pickle as pk
import numpy as np
from torch import nn
from alignn.data import get_train_val_loaders
from alignn.config import TrainingConfig
from alignn.models.alignn_atomwise import ALIGNNAtomWise
from jarvis.db.jsonutils import dumpjson
import json
import pprint
import os
import warnings
import time
from sklearn.metrics import roc_auc_score

warnings.filterwarnings("ignore", category=RuntimeWarning)
torch.set_default_dtype(torch.float32)


def activated_output_transform(output):
    """Exponentiate output."""
    y_pred, y = output
    y_pred = torch.exp(y_pred)
    y_pred = y_pred[:, 1]
    return y_pred, y


def make_standard_scalar_and_pca(output):
    """Use standard scalar and PCS for multi-output data."""
    sc = pk.load(open(os.path.join(tmp_output_dir, "sc.pkl"), "rb"))
    y_pred, y = output
    y_pred = torch.tensor(
        sc.transform(y_pred.cpu().numpy()), device=y_pred.device
    )
    y = torch.tensor(sc.transform(y.cpu().numpy()), device=y.device)
    # pc = pk.load(open("pca.pkl", "rb"))
    # y_pred = torch.tensor(pc.transform(y_pred), device=device)
    # y = torch.tensor(pc.transform(y), device=device)
    # y_pred = torch.tensor(pca_sc.inverse_transform(y_pred),device=device)
    # y = torch.tensor(pca_sc.inverse_transform(y),device=device)
    # print (y.shape,y_pred.shape)
    return y_pred, y


def thresholded_output_transform(output):
    """Round off output."""
    y_pred, y = output
    y_pred = torch.round(torch.exp(y_pred))
    # print ('output',y_pred)
    return y_pred, y


def group_decay(model):
    """Omit weight decay from bias and batchnorm params."""
    decay, no_decay = [], []

    for name, p in model.named_parameters():
        if "bias" in name or "bn" in name or "norm" in name:
            no_decay.append(p)
        else:
            decay.append(p)

    return [
        {"params": decay},
        {"params": no_decay, "weight_decay": 0},
    ]


def setup_optimizer(params, config: TrainingConfig):
    """Set up optimizer for param groups."""
    if config.optimizer == "adamw":
        optimizer = torch.optim.AdamW(
            params,
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
        )
    elif config.optimizer == "sgd":
        optimizer = torch.optim.SGD(
            params,
            lr=config.learning_rate,
            momentum=0.9,
            weight_decay=config.weight_decay,
        )
    return optimizer


def train_dgl(
    config: Union[TrainingConfig, Dict[str, Any]],
    model: nn.Module = None,
    # checkpoint_dir: Path = Path("./"),
    train_val_test_loaders=[],
    # log_tensorboard: bool = False,
):
    """Training entry point for DGL networks.

    `config` should conform to alignn.conf.TrainingConfig, and
    if passed as a dict with matching keys, pydantic validation is used
    """
    print(config)
    if type(config) is dict:
        try:
            print(config)
            config = TrainingConfig(**config)
        except Exception as exp:
            print("Check", exp)

    if not os.path.exists(config.output_dir):
        os.makedirs(config.output_dir)
    checkpoint_dir = os.path.join(config.output_dir)
    deterministic = False
    classification = False
    print("config:")
    tmp = config.dict()
    f = open(os.path.join(config.output_dir, "config.json"), "w")
    f.write(json.dumps(tmp, indent=4))
    f.close()
    global tmp_output_dir
    tmp_output_dir = config.output_dir
    pprint.pprint(tmp)  # , sort_dicts=False)
    if config.classification_threshold is not None:
        classification = True

    line_graph = False
    if config.model.alignn_layers > 0:
        line_graph = True
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
        )
    else:
        train_loader = train_val_test_loaders[0]
        val_loader = train_val_test_loaders[1]
        test_loader = train_val_test_loaders[2]
        prepare_batch = train_val_test_loaders[3]
    device = "cpu"
    if torch.cuda.is_available():
        device = torch.device("cuda")
    prepare_batch = partial(prepare_batch, device=device)
    if classification:
        config.model.classification = True
    _model = {
        "alignn_atomwise": ALIGNNAtomWise,
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
    net.to(device)
    if config.data_parallel and torch.cuda.device_count() > 1:
        # For multi-GPU training make data_parallel:true in config.json file
        device_ids = [cid for cid in range(torch.cuda.device_count())]
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        net = torch.nn.DataParallel(net, device_ids=device_ids).cuda()
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

    if config.model.name == "alignn_atomwise":

        def get_batch_errors(dat=[]):
            """Get errors for samples."""
            target_out = []
            pred_out = []
            grad = []
            atomw = []
            stress = []
            mean_out = 0
            mean_atom = 0
            mean_grad = 0
            mean_stress = 0
            # natoms_batch=False
            # print ('lendat',len(dat))
            for i in dat:
                if i["target_grad"]:
                    # if config.normalize_graph_level_loss:
                    #      natoms_batch = 0
                    for m, n in zip(i["target_grad"], i["pred_grad"]):
                        x = np.abs(np.array(m) - np.array(n))
                        grad.append(np.mean(x))
                        # if config.normalize_graph_level_loss:
                        #     natoms_batch += np.array(i["pred_grad"]).shape[0]
                if i["target_out"]:
                    for j, k in zip(i["target_out"], i["pred_out"]):
                        # if config.normalize_graph_level_loss and
                        # natoms_batch:
                        #   j=j/natoms_batch
                        #   k=k/natoms_batch
                        # if config.normalize_graph_level_loss and
                        # not natoms_batch:
                        # tmp = 'Add above in atomwise if not train grad.'
                        #   raise ValueError(tmp)

                        target_out.append(j)
                        pred_out.append(k)
                if i["target_stress"]:
                    for p, q in zip(i["target_stress"], i["pred_stress"]):
                        x = np.abs(np.array(p) - np.array(q))
                        stress.append(np.mean(x))
                if i["target_atomwise_pred"]:
                    for m, n in zip(
                        i["target_atomwise_pred"], i["pred_atomwise_pred"]
                    ):
                        x = np.abs(np.array(m) - np.array(n))
                        atomw.append(np.mean(x))
            if "target_out" in i:
                # if i["target_out"]:
                target_out = np.array(target_out)
                pred_out = np.array(pred_out)
                # print('target_out',target_out,target_out.shape)
                # print('pred_out',pred_out,pred_out.shape)
                if classification:
                    mean_out = log_loss(target_out, pred_out)
                else:
                    mean_out = mean_absolute_error(target_out, pred_out)
            if "target_stress" in i:
                # if i["target_stress"]:
                mean_stress = np.array(stress).mean()
            if "target_grad" in i:
                # if i["target_grad"]:
                mean_grad = np.array(grad).mean()
            if "target_atomwise_pred" in i:
                # if i["target_atomwise_pred"]:
                mean_atom = np.array(atomw).mean()
            # print ('natoms_batch',natoms_batch)
            # if natoms_batch!=0:
            #   mean_out = mean_out/natoms_batch
            # print ('dat',dat)
            return mean_out, mean_atom, mean_grad, mean_stress

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
            train_result = []
            # for dats in train_loader:
            for dats, jid in zip(train_loader, train_loader.dataset.ids):
                info = {}
                # info["id"] = jid
                optimizer.zero_grad()
                result = net([dats[0].to(device), dats[1].to(device)])
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
                    # print('result["out"]',result["out"])
                    # print('dats[2]',dats[2])
                    loss1 = config.model.graphwise_weight * criterion(
                        result["out"], dats[2].to(device)
                    )
                    info["target_out"] = dats[2].cpu().numpy().tolist()
                    info["pred_out"] = (
                        result["out"].cpu().detach().numpy().tolist()
                    )
                    # graphlevel_loss += np.mean(
                    #    np.abs(
                    #        dats[2].cpu().numpy()
                    #        - result["out"].cpu().detach().numpy()
                    #    )
                    # )
                    # print("target_out", info["target_out"][0])
                    # print("pred_out", info["pred_out"][0])
                # print(
                #    "config.model.atomwise_output_features",
                #    config.model.atomwise_output_features,
                # )
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
                    # atomlevel_loss += np.mean(
                    #    np.abs(
                    #        dats[0].ndata["atomwise_target"].cpu().numpy()
                    #        - result["atomwise_pred"].cpu().detach().numpy()
                    #    )
                    # )

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
                    # gradlevel_loss += np.mean(
                    #    np.abs(
                    #        dats[0].ndata["atomwise_grad"].cpu().numpy()
                    #        - result["grad"].cpu().detach().numpy()
                    #    )
                    # )
                    # print("target_grad", info["target_grad"][0])
                    # print("pred_grad", info["pred_grad"][0])
                if config.model.stresswise_weight != 0:
                    # print(
                    #    'result["stress"]',
                    #    result["stresses"],
                    #    result["stresses"].shape,
                    # )
                    # print(
                    #    'dats[0].ndata["stresses"]',
                    #    torch.cat(tuple(dats[0].ndata["stresses"])),
                    #    dats[0].ndata["stresses"].shape,
                    # )  # ,torch.cat(dats[0].ndata["stresses"]),
                    # torch.cat(dats[0].ndata["stresses"]).shape)
                    # print('result["stresses"]',result["stresses"],result["stresses"].shape)
                    # print(dats[0].ndata["stresses"],dats[0].ndata["stresses"].shape)
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
                    # print("target_stress", info["target_stress"][0])
                    # print("pred_stress", info["pred_stress"][0])
                train_result.append(info)
                loss = loss1 + loss2 + loss3 + loss4
                loss.backward()
                optimizer.step()
                # optimizer.zero_grad() #never
                running_loss += loss.item()
            mean_out, mean_atom, mean_grad, mean_stress = get_batch_errors(
                train_result
            )
            # dumpjson(filename="Train_results.json", data=train_result)
            scheduler.step()
            train_final_time = time.time()
            train_ep_time = train_final_time - train_init_time
            print(
                "TrainLoss",
                "Epoch",
                e,
                "total",
                running_loss,
                "out",
                mean_out,
                "atom",
                mean_atom,
                "grad",
                mean_grad,
                "stress",
                mean_stress,
                "time",
                train_ep_time,
            )
            history_train.append([mean_out, mean_atom, mean_grad, mean_stress])
            dumpjson(
                filename=os.path.join(config.output_dir, "history_train.json"),
                data=history_train,
            )
            val_loss = 0
            val_result = []
            # for dats in val_loader:
            for dats, jid in zip(val_loader, val_loader.dataset.ids):
                info = {}
                info["id"] = jid
                optimizer.zero_grad()
                result = net([dats[0].to(device), dats[1].to(device)])
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
                        result["out"], dats[2].to(device)
                    )
                    info["target_out"] = dats[2].cpu().numpy().tolist()
                    info["pred_out"] = (
                        result["out"].cpu().detach().numpy().tolist()
                    )

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
                    # loss4 = config.model.stresswise_weight * criterion(
                    #    result["stress"].to(device),
                    #    dats[0].ndata["stresses"][0].to(device),
                    # )
                    loss4 = config.model.stresswise_weight * criterion(
                        # torch.flatten(result["stress"].to(device)),
                        # (dats[0].ndata["stresses"]).to(device),
                        # torch.flatten(dats[0].ndata["stresses"]).to(device),
                        # torch.flatten(torch.cat(dats[0].ndata["stresses"])).to(device),
                        # dats[0].ndata["stresses"][0].to(device),
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
                loss = loss1 + loss2 + loss3 + loss4
                val_result.append(info)
                val_loss += loss.item()
            mean_out, mean_atom, mean_grad, mean_stress = get_batch_errors(
                val_result
            )
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
            print(
                "ValLoss",
                "Epoch",
                e,
                "total",
                val_loss,
                "out",
                mean_out,
                "atom",
                mean_atom,
                "grad",
                mean_grad,
                "stress",
                mean_stress,
                saving_msg,
            )
            history_val.append([mean_out, mean_atom, mean_grad, mean_stress])
            dumpjson(
                filename=os.path.join(config.output_dir, "history_val.json"),
                data=history_val,
            )

        test_loss = 0
        test_result = []
        for dats, jid in zip(test_loader, test_loader.dataset.ids):
            # for dats in test_loader:
            info = {}
            info["id"] = jid
            optimizer.zero_grad()
            # print('dats[0]',dats[0])
            # print('test_loader',test_loader)
            # print('test_loader.dataset.ids',test_loader.dataset.ids)
            result = net([dats[0].to(device), dats[1].to(device)])
            loss1 = 0  # Such as energy
            loss2 = 0  # Such as bader charges
            loss3 = 0  # Such as forces
            loss4 = 0  # Such as stresses
            if config.model.output_features is not None and not classification:
                # print('result["out"]',result["out"])
                # print('dats[2]',dats[2])
                loss1 = config.model.graphwise_weight * criterion(
                    result["out"], dats[2].to(device)
                )
                info["target_out"] = dats[2].cpu().numpy().tolist()
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
                    # torch.flatten(result["stress"].to(device)),
                    # (dats[0].ndata["stresses"]).to(device),
                    # torch.flatten(dats[0].ndata["stresses"]).to(device),
                    result["stresses"].to(device),
                    torch.cat(tuple(dats[0].ndata["stresses"])).to(device),
                    # torch.flatten(torch.cat(dats[0].ndata["stresses"])).to(device),
                    # dats[0].ndata["stresses"][0].to(device),
                )
                # loss4 = config.model.stresswise_weight * criterion(
                #    result["stress"][0].to(device),
                #    dats[0].ndata["stresses"].to(device),
                # )
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

    if config.write_predictions and classification:
        best_model.eval()
        # net.eval()
        f = open(
            os.path.join(config.output_dir, "prediction_results_test_set.csv"),
            "w",
        )
        f.write("id,target,prediction\n")
        targets = []
        predictions = []
        with torch.no_grad():
            ids = test_loader.dataset.ids  # [test_loader.dataset.indices]
            for dat, id in zip(test_loader, ids):
                g, lg, target = dat
                out_data = net([g.to(device), lg.to(device)])["out"]
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
                out_data = net([g.to(device), lg.to(device)])["out"]
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
            os.path.join(config.output_dir, "prediction_results_test_set.csv"),
            "w",
        )
        f.write("id,target,prediction\n")
        targets = []
        predictions = []
        with torch.no_grad():
            ids = test_loader.dataset.ids  # [test_loader.dataset.indices]
            for dat, id in zip(test_loader, ids):
                g, lg, target = dat
                out_data = net([g.to(device), lg.to(device)])["out"]
                out_data = out_data.cpu().numpy().tolist()
                if config.standard_scalar_and_pca:
                    sc = pk.load(
                        open(os.path.join(tmp_output_dir, "sc.pkl"), "rb")
                    )
                    out_data = sc.transform(np.array(out_data).reshape(-1, 1))[
                        0
                    ][0]
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
                out_data = net([g.to(device), lg.to(device)])["out"]
                out_data = out_data.cpu().numpy().tolist()
                if config.standard_scalar_and_pca:
                    sc = pk.load(
                        open(os.path.join(tmp_output_dir, "sc.pkl"), "rb")
                    )
                    out_data = sc.transform(np.array(out_data).reshape(-1, 1))[
                        0
                    ][0]
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


if __name__ == "__main__":
    config = TrainingConfig(
        random_seed=123, epochs=10, n_train=32, n_val=32, batch_size=16
    )
    history = train_dgl(config)
