"""Ignite training script.

from the repository root, run
`PYTHONPATH=$PYTHONPATH:. python alignn/train.py`
then `tensorboard --logdir tb_logs/test` to monitor results...
"""

from functools import partial

# from pathlib import Path
from typing import Any, Dict, Union
import ignite
import torch
import random
from ignite.contrib.handlers import TensorboardLogger
from sklearn.metrics import mean_absolute_error

try:
    from ignite.contrib.handlers.stores import EpochOutputStore

    # For different version of pytorch-ignite
except Exception:
    from ignite.handlers.stores import EpochOutputStore

    pass
from ignite.handlers import EarlyStopping
from ignite.contrib.handlers.tensorboard_logger import (
    global_step_from_engine,
)
from ignite.contrib.handlers.tqdm_logger import ProgressBar
from ignite.engine import (
    Events,
    create_supervised_evaluator,
    create_supervised_trainer,
)
from ignite.contrib.metrics import ROC_AUC, RocCurve
from ignite.metrics import (
    Accuracy,
    Precision,
    Recall,
    ConfusionMatrix,
)
import pickle as pk
import numpy as np
from ignite.handlers import Checkpoint, DiskSaver, TerminateOnNan
from ignite.metrics import Loss, MeanAbsoluteError
from torch import nn
from alignn.data import get_train_val_loaders
from alignn.config import TrainingConfig
from alignn.models.alignn import ALIGNN
from alignn.models.alignn_atomwise import ALIGNNAtomWise

# from alignn import models
# from alignn.models.alignn_layernorm import ALIGNN as ALIGNN_LN
# from alignn.models.modified_cgcnn import CGCNN
# from alignn.models.dense_alignn import DenseALIGNN
# from alignn.models.densegcn import DenseGCN
# from alignn.models.icgcnn import iCGCNN
# from alignn.models.alignn_cgcnn import ACGCNN
from jarvis.db.jsonutils import dumpjson
import json
import pprint

# from accelerate import Accelerator
import os
import warnings

warnings.filterwarnings("ignore", category=RuntimeWarning)
# from sklearn.decomposition import PCA, KernelPCA
# from sklearn.preprocessing import StandardScaler

# torch config
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
    import os

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
    if config.random_seed is not None:
        deterministic = True
        ignite.utils.manual_seed(config.random_seed)

    line_graph = False
    alignn_models = {
        "alignn",
        # "dense_alignn",
        # "alignn_cgcnn",
        # "alignn_layernorm",
    }
    # if config.model.name == "clgn":
    #    line_graph = True
    # if config.model.name == "cgcnn":
    #    line_graph = True
    # if config.model.name == "icgcnn":
    #    line_graph = True
    if config.model.name in alignn_models and config.model.alignn_layers > 0:
        line_graph = True
    # print ('output_dir train', config.output_dir)
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
            # ) = data.get_train_val_loaders(
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
    if config.distributed:
        print(
            "Using Accelerator, currently experimental, use at your own risk."
        )
        from accelerate import Accelerator

        accelerator = Accelerator()
        device = accelerator.device
    prepare_batch = partial(prepare_batch, device=device)
    if classification:
        config.model.classification = True
    # define network, optimizer, scheduler
    _model = {
        # "cgcnn": CGCNN,
        # "icgcnn": iCGCNN,
        # "densegcn": DenseGCN,
        "alignn": ALIGNN,
        "alignn_atomwise": ALIGNNAtomWise,
        # "dense_alignn": DenseALIGNN,
        # "alignn_cgcnn": ACGCNN,
        # "alignn_layernorm": ALIGNN_LN,
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
    if config.data_parallel and torch.cuda.device_count() > 1:
        # For multi-GPU training make data_parallel:true in config.json file
        device_ids = [cid for cid in range(torch.cuda.device_count())]
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        net = torch.nn.DataParallel(net, device_ids=device_ids).cuda()
    net.to(device)
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
        # criterion = nn.MSELoss()
        params = group_decay(net)
        optimizer = setup_optimizer(params, config)
        # optimizer = torch.optim.Adam(net.parameters(), lr=0.001)
        if config.distributed:
            from accelerate import Accelerator

            accelerator = Accelerator()
            print("Using Accelerator device", accelerator.device)

            device = accelerator.device
            # print('model',net)
            net.to(device)
            train_loader, val_loader, model, optimizer = accelerator.prepare(
                train_loader, val_loader, net, optimizer
            )
        history_train = []
        history_val = []
        for e in range(config.epochs):
            # optimizer.zero_grad()
            running_loss = 0
            train_result = []
            # for dats in train_loader:
            for dats, jid in zip(train_loader, train_loader.dataset.ids):
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
                if config.distributed:
                    from accelerate import Accelerator

                    accelerator = Accelerator()
                    accelerator.backward(loss)
                else:
                    loss.backward()
                optimizer.step()
                # optimizer.zero_grad() #never
                running_loss += loss.item()
            mean_out, mean_atom, mean_grad, mean_stress = get_batch_errors(
                train_result
            )
            # dumpjson(filename="Train_results.json", data=train_result)
            scheduler.step()
            if config.distributed:
                from accelerate import Accelerator

                accelerator = Accelerator()
                accelerator.print(
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
                )

            else:
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
            if val_loss < best_loss:
                best_loss = val_loss
                best_model_name = "best_model.pt"
                torch.save(
                    net.state_dict(),
                    os.path.join(config.output_dir, best_model_name),
                )
                if config.distributed:
                    from accelerate import Accelerator

                    accelerator = Accelerator()
                    accelerator.print("Saving data for epoch:", e)
                else:
                    print("Saving data for epoch:", e)
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
            if config.distributed:
                from accelerate import Accelerator

                accelerator = Accelerator()
                accelerator.print(
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
                )
            else:
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
            if config.model.output_features is not None:
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
            test_loss += loss.item()
        if config.distributed:
            from accelerate import Accelerator

            accelerator = Accelerator()
            accelerator.print("TestLoss", e, test_loss)
        else:
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
        return test_result

    if config.distributed:
        import torch.distributed as dist
        import os

        print()
        print()
        print()
        gpus = torch.cuda.device_count()
        print("Using DistributedDataParallel !!!", gpus)

        def setup(rank, world_size):
            os.environ["MASTER_ADDR"] = "localhost"
            os.environ["MASTER_PORT"] = "12355"

            # initialize the process group
            dist.init_process_group("gloo", rank=rank, world_size=world_size)

        def cleanup():
            dist.destroy_process_group()

        setup(2, 2)
        local_rank = [0, 1]
        # net=torch.nn.parallel.DataParallel(net
        # ,device_ids=[local_rank, ],output_device=local_rank)
        net = torch.nn.parallel.DistributedDataParallel(
            net,
            device_ids=[
                local_rank,
            ],
            output_device=local_rank,
        )
        print()
        print()
        print()
        # )  # ,device_ids=[local_rank, ],output_device=local_rank)
    """
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

    """
    # select configured loss function
    criteria = {
        "mse": nn.MSELoss(),
        "l1": nn.L1Loss(),
        "poisson": nn.PoissonNLLLoss(log_input=False, full=True),
        # "zig": models.modified_cgcnn.ZeroInflatedGammaLoss(),
    }
    criterion = criteria[config.criterion]

    # set up training engine and evaluators
    metrics = {"loss": Loss(criterion), "mae": MeanAbsoluteError()}
    if config.model.output_features > 1 and config.standard_scalar_and_pca:
        # metrics = {"loss": Loss(criterion), "mae": MeanAbsoluteError()}
        metrics = {
            "loss": Loss(
                criterion, output_transform=make_standard_scalar_and_pca
            ),
            "mae": MeanAbsoluteError(
                output_transform=make_standard_scalar_and_pca
            ),
        }

    if config.criterion == "zig":

        def zig_prediction_transform(x):
            output, y = x
            return criterion.predict(output), y

        metrics = {
            "loss": Loss(criterion),
            "mae": MeanAbsoluteError(
                output_transform=zig_prediction_transform
            ),
        }

    if classification:
        criterion = nn.NLLLoss()

        metrics = {
            "accuracy": Accuracy(
                output_transform=thresholded_output_transform
            ),
            "precision": Precision(
                output_transform=thresholded_output_transform
            ),
            "recall": Recall(output_transform=thresholded_output_transform),
            "rocauc": ROC_AUC(output_transform=activated_output_transform),
            "roccurve": RocCurve(output_transform=activated_output_transform),
            "confmat": ConfusionMatrix(
                output_transform=thresholded_output_transform, num_classes=2
            ),
        }
    trainer = create_supervised_trainer(
        net,
        optimizer,
        criterion,
        prepare_batch=prepare_batch,
        device=device,
        deterministic=deterministic,
        # output_transform=make_standard_scalar_and_pca,
    )

    evaluator = create_supervised_evaluator(
        net,
        metrics=metrics,
        prepare_batch=prepare_batch,
        device=device,
        # output_transform=make_standard_scalar_and_pca,
    )

    train_evaluator = create_supervised_evaluator(
        net,
        metrics=metrics,
        prepare_batch=prepare_batch,
        device=device,
        # output_transform=make_standard_scalar_and_pca,
    )

    # ignite event handlers:
    trainer.add_event_handler(Events.EPOCH_COMPLETED, TerminateOnNan())

    # apply learning rate scheduler
    trainer.add_event_handler(
        Events.ITERATION_COMPLETED, lambda engine: scheduler.step()
    )

    if config.write_checkpoint:
        # model checkpointing
        to_save = {
            "model": net,
            "optimizer": optimizer,
            "lr_scheduler": scheduler,
            "trainer": trainer,
        }
        if classification:

            def cp_score(engine):
                """Higher accuracy is better."""
                return engine.state.metrics["accuracy"]

        else:

            def cp_score(engine):
                """Lower MAE is better."""
                return -engine.state.metrics["mae"]

        # save last two epochs
        evaluator.add_event_handler(
            Events.EPOCH_COMPLETED,
            Checkpoint(
                to_save,
                DiskSaver(
                    checkpoint_dir, create_dir=True, require_empty=False
                ),
                n_saved=2,
                global_step_transform=lambda *_: trainer.state.epoch,
            ),
        )
        # save best model
        evaluator.add_event_handler(
            Events.EPOCH_COMPLETED,
            Checkpoint(
                to_save,
                DiskSaver(
                    checkpoint_dir, create_dir=True, require_empty=False
                ),
                filename_pattern="best_model.{ext}",
                n_saved=1,
                global_step_transform=lambda *_: trainer.state.epoch,
                score_function=cp_score,
            ),
        )
    if config.progress:
        pbar = ProgressBar()
        pbar.attach(trainer, output_transform=lambda x: {"loss": x})
        # pbar.attach(evaluator,output_transform=lambda x: {"mae": x})

    history = {
        "train": {m: [] for m in metrics.keys()},
        "validation": {m: [] for m in metrics.keys()},
    }

    if config.store_outputs:
        # log_results handler will save epoch output
        # in history["EOS"]
        eos = EpochOutputStore()
        eos.attach(evaluator)
        train_eos = EpochOutputStore()
        train_eos.attach(train_evaluator)

    # collect evaluation performance
    @trainer.on(Events.EPOCH_COMPLETED)
    def log_results(engine):
        """Print training and validation metrics to console."""
        train_evaluator.run(train_loader)
        evaluator.run(val_loader)

        tmetrics = train_evaluator.state.metrics
        vmetrics = evaluator.state.metrics
        for metric in metrics.keys():
            tm = tmetrics[metric]
            vm = vmetrics[metric]
            if metric == "roccurve":
                tm = [k.tolist() for k in tm]
                vm = [k.tolist() for k in vm]
            if isinstance(tm, torch.Tensor):
                tm = tm.cpu().numpy().tolist()
                vm = vm.cpu().numpy().tolist()

            history["train"][metric].append(tm)
            history["validation"][metric].append(vm)

        # for metric in metrics.keys():
        #    history["train"][metric].append(tmetrics[metric])
        #    history["validation"][metric].append(vmetrics[metric])

        if config.store_outputs:
            history["EOS"] = eos.data
            history["trainEOS"] = train_eos.data
            dumpjson(
                filename=os.path.join(config.output_dir, "history_val.json"),
                data=history["validation"],
            )
            dumpjson(
                filename=os.path.join(config.output_dir, "history_train.json"),
                data=history["train"],
            )
        if config.progress:
            pbar = ProgressBar()
            if not classification:
                pbar.log_message(f"Val_MAE: {vmetrics['mae']:.4f}")
                pbar.log_message(f"Train_MAE: {tmetrics['mae']:.4f}")
            else:
                pbar.log_message(f"Train ROC AUC: {tmetrics['rocauc']:.4f}")
                pbar.log_message(f"Val ROC AUC: {vmetrics['rocauc']:.4f}")

    if config.n_early_stopping is not None:
        # early stopping if no improvement (improvement = higher score)
        if classification:

            def es_score(engine):
                """Higher accuracy is better."""
                return engine.state.metrics["accuracy"]

        else:

            def es_score(engine):
                """Lower MAE is better."""
                return -engine.state.metrics["mae"]

        es_handler = EarlyStopping(
            patience=config.n_early_stopping,
            score_function=es_score,
            trainer=trainer,
        )
        evaluator.add_event_handler(Events.EPOCH_COMPLETED, es_handler)

    # optionally log results to tensorboard
    if config.log_tensorboard:
        tb_logger = TensorboardLogger(
            log_dir=os.path.join(config.output_dir, "tb_logs", "test")
        )
        for tag, evaluator in [
            ("training", train_evaluator),
            ("validation", evaluator),
        ]:
            tb_logger.attach_output_handler(
                evaluator,
                event_name=Events.EPOCH_COMPLETED,
                tag=tag,
                metric_names=["loss", "mae"],
                global_step_transform=global_step_from_engine(trainer),
            )

    # train the model!
    trainer.run(train_loader, max_epochs=config.epochs)

    if config.log_tensorboard:
        test_loss = evaluator.state.metrics["loss"]
        tb_logger.writer.add_hparams(config, {"hparam/test_loss": test_loss})
        tb_logger.close()
    if config.write_predictions and classification:
        net.eval()
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
                out_data = net([g.to(device), lg.to(device)])
                # out_data = torch.exp(out_data.cpu())
                top_p, top_class = torch.topk(torch.exp(out_data), k=1)
                target = int(target.cpu().numpy().flatten().tolist()[0])

                f.write("%s, %d, %d\n" % (id, (target), (top_class)))
                targets.append(target)
                predictions.append(
                    top_class.cpu().numpy().flatten().tolist()[0]
                )
        f.close()
        from sklearn.metrics import roc_auc_score

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
        net.eval()
        mem = []
        with torch.no_grad():
            ids = test_loader.dataset.ids  # [test_loader.dataset.indices]
            for dat, id in zip(test_loader, ids):
                g, lg, target = dat
                out_data = net([g.to(device), lg.to(device)])
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
    ):
        net.eval()
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
                out_data = net([g.to(device), lg.to(device)])
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
        if config.store_outputs and not classification:
            # save training targets and predictions here
            # TODO: Add IDs
            resultsfile = os.path.join(
                config.output_dir, "prediction_results_train_set.csv"
            )

            target_vals, predictions = [], []

            for tgt, pred in history["trainEOS"]:
                target_vals.append(tgt.cpu().numpy().tolist())
                predictions.append(pred.cpu().numpy().tolist())

            target_vals = np.array(target_vals, dtype="float").flatten()
            predictions = np.array(predictions, dtype="float").flatten()

            with open(resultsfile, "w") as f:
                print("target,prediction", file=f)
                for target_val, predicted_val in zip(target_vals, predictions):
                    print(f"{target_val}, {predicted_val}", file=f)

    # TODO: Fix IDs for train loader
    """
    if config.write_train_predictions:
        net.eval()
        f = open("train_prediction_results.csv", "w")
        f.write("id,target,prediction\n")
        with torch.no_grad():
            ids = train_loader.dataset.dataset.ids[
                train_loader.dataset.indices
            ]
            print("lens", len(ids), len(train_loader.dataset.dataset))
            x = []
            y = []

            for dat, id in zip(train_loader, ids):
                g, lg, target = dat
                out_data = net([g.to(device), lg.to(device)])
                out_data = out_data.cpu().numpy().tolist()
                target = target.cpu().numpy().flatten().tolist()
                for i, j in zip(out_data, target):
                    x.append(i)
                    y.append(j)
            for i, j, k in zip(ids, x, y):
                f.write("%s, %6f, %6f\n" % (i, j, k))
        f.close()

    """
    return history


if __name__ == "__main__":
    config = TrainingConfig(
        random_seed=123, epochs=10, n_train=32, n_val=32, batch_size=16
    )
    history = train_dgl(config, progress=True)
