#!/usr/bin/env python

"""Module to train for a folder with formatted dataset."""
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
import numpy as np
import argparse
import sys
import json
import random
import time
import zipfile
import csv
from alignn.data import get_train_val_loaders
from alignn.config import TrainingConfig
from alignn.models.alignn_atomwise import ALIGNNAtomWise, ALIGNNAtomWiseConfig
from jarvis.db.jsonutils import loadjson, dumpjson
from jarvis.core.atoms import Atoms
from ase.stress import voigt_6_to_full_3x3_stress
from torch.utils.data import DataLoader
import pprint

device = "cpu"
if torch.cuda.is_available():
    device = torch.device("cuda")


def setup(rank=0, world_size=0, port="12356"):
    """Set up multi GPU rank."""
    if port == "":
        port = str(random.randint(10000, 99999))
    if world_size > 1:
        os.environ["MASTER_ADDR"] = "localhost"
        os.environ["MASTER_PORT"] = port
        dist.init_process_group("nccl", rank=rank, world_size=world_size)
        torch.cuda.set_device(rank)


def cleanup(world_size):
    """Clean up distributed process."""
    if world_size > 1:
        dist.destroy_process_group()


def group_decay(model):
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
    if config.optimizer == "adamw":
        return torch.optim.AdamW(
            params, lr=config.learning_rate, weight_decay=config.weight_decay
        )
    elif config.optimizer == "sgd":
        return torch.optim.SGD(
            params,
            lr=config.learning_rate,
            momentum=0.9,
            weight_decay=config.weight_decay,
        )


class Trainer:
    def __init__(
        self,
        config: TrainingConfig,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        optimizer: torch.optim.Optimizer,
        rank: int,
        world_size: int,
        device: torch.device,
    ) -> None:
        self.config = config
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer = optimizer
        self.rank = rank
        self.device = device
        self.model = model.to(self.device)
        self.scheduler = self._setup_scheduler(optimizer)
        self.history_train = []
        self.history_val = []
        self.best_val_loss = float("inf")
        if world_size > 1:
            self.model = torch.nn.parallel.DistributedDataParallel(
                model, device_ids=[rank], find_unused_parameters=True
            )

    def _setup_scheduler(self, optimizer):
        if self.config.scheduler == "none":
            return torch.optim.lr_scheduler.LambdaLR(
                optimizer, lambda epoch: 1.0
            )
        elif self.config.scheduler == "onecycle":
            steps_per_epoch = len(self.train_loader)
            return torch.optim.lr_scheduler.OneCycleLR(
                optimizer,
                max_lr=self.config.learning_rate,
                epochs=self.config.epochs,
                steps_per_epoch=steps_per_epoch,
                pct_start=0.3,
            )
        elif self.config.scheduler == "step":
            return torch.optim.lr_scheduler.StepLR(optimizer)

    def _run_batch(self, batch):
        self.optimizer.zero_grad()
        loss = self._compute_loss(batch)
        loss.backward()
        self.optimizer.step()
        return loss.item()

    def _compute_loss(self, batch):
        # print(batch,len(batch))
        dats = batch
        # dats, jid = batch
        if self.config.model.alignn_layers > 0:

            result = self.model(
                [dats[0].to(self.device), dats[1].to(self.device)]
            )
        else:
            result = self.model(dats[0].to(self.device))

        loss1 = self.config.model.graphwise_weight * F.l1_loss(
            result["out"], dats[-1].to(self.device)
        )
        loss2 = 0
        if (
            self.config.model.atomwise_output_features > 0
            and self.config.model.atomwise_weight != 0
        ):

            loss2 = self.config.model.atomwise_weight * F.l1_loss(
                result["atomwise_pred"].to(self.device),
                dats[0].ndata["atomwise_target"].to(self.device),
            )
        loss3 = (
            self.config.model.gradwise_weight
            * F.l1_loss(
                result["grad"].to(self.device),
                dats[0].ndata["atomwise_grad"].to(self.device),
            )
            if self.config.model.calculate_gradient
            else 0
        )
        loss4 = (
            self.config.model.stresswise_weight
            * F.l1_loss(
                result["stresses"].to(self.device),
                torch.cat(tuple(dats[0].ndata["stresses"])).to(self.device),
            )
            if self.config.model.stresswise_weight != 0
            else 0
        )
        # print('loss1 , loss2 , loss3 , loss4',loss1 , loss2 , loss3 , loss4)
        return loss1 + loss2 + loss3 + loss4

    def _run_epoch(self, epoch):
        self.model.train()
        train_loss = 0
        for batch in self.train_loader:
            train_loss += self._run_batch(batch)
        train_loss /= len(self.train_loader)
        self.history_train.append(train_loss)

        self.model.eval()
        val_loss = 0
        # """
        # with torch.no_grad():
        for batch in self.val_loader:
            val_loss += self._compute_loss(batch).item()
        val_loss /= len(self.val_loader)
        # """
        self.history_val.append(val_loss)

        self.scheduler.step()

        print(
            f"[GPU{self.rank}] Epoch {epoch} | Training Loss: {train_loss} | Validation Loss: {val_loss}"
        )
        save_every = 1
        if self.rank == 0:
            dumpjson(
                filename=os.path.join(
                    self.config.output_dir, "history_train.json"
                ),
                data=self.history_train,
            )
            dumpjson(
                filename=os.path.join(
                    self.config.output_dir, "history_val.json"
                ),
                data=self.history_val,
            )

            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self._save_checkpoint(epoch, best=True)

            if epoch % save_every == 0:
                # if epoch % self.config.save_every == 0:
                self._save_checkpoint(epoch)

    def _save_checkpoint(self, epoch, best=False):
        checkpoint_name = f"best_model.pt" if best else f"current_model.pt"
        # checkpoint_name = f"best_model_epoch_{epoch}.pt" if best else f"checkpoint_epoch_{epoch}.pt"
        torch.save(
            self.model.state_dict(),
            os.path.join(self.config.output_dir, checkpoint_name),
        )
        print(f"Epoch {epoch} | Model checkpoint saved as {checkpoint_name}")

    def train(self):
        for epoch in range(self.config.epochs):
            self._run_epoch(epoch)


def train_for_folder(
    rank=0,
    world_size=0,
    root_dir="examples/sample_data",
    config_name="config.json",
    classification_threshold=None,
    batch_size=None,
    epochs=None,
    id_key="jid",
    target_key="total_energy",
    atomwise_key="forces",
    gradwise_key="forces",
    stresswise_key="stresses",
    file_format="poscar",
    restart_model_path=None,
    restart_model_name="current_model.pt",
    output_dir=None,
):
    """Train for a folder."""
    setup(rank=rank, world_size=world_size)
    print("root_dir", root_dir)
    id_prop_json = os.path.join(root_dir, "id_prop.json")
    id_prop_json_zip = os.path.join(root_dir, "id_prop.json.zip")
    id_prop_csv = os.path.join(root_dir, "id_prop.csv")
    id_prop_csv_file = False
    multioutput = False

    if os.path.exists(id_prop_json_zip):
        dat = json.loads(
            zipfile.ZipFile(id_prop_json_zip).read("id_prop.json")
        )
    elif os.path.exists(id_prop_json):
        dat = loadjson(os.path.join(root_dir, "id_prop.json"))
    elif os.path.exists(id_prop_csv):
        id_prop_csv_file = True
        with open(id_prop_csv, "r") as f:
            reader = csv.reader(f)
            dat = [row for row in reader]
        print("id_prop_csv_file exists", id_prop_csv_file)
    else:
        print("Check dataset file.")

    config_dict = loadjson(config_name)
    config = TrainingConfig(**config_dict)
    pprint.pprint(config_dict)
    if classification_threshold is not None:
        config.classification_threshold = float(classification_threshold)
    if output_dir is not None:
        config.output_dir = output_dir
    if batch_size is not None:
        config.batch_size = int(batch_size)
    if epochs is not None:
        config.epochs = int(epochs)

    train_grad = (
        config.model.calculate_gradient and config.model.gradwise_weight != 0
    )
    train_stress = (
        config.model.calculate_gradient and config.model.stresswise_weight != 0
    )
    train_atom = config.model.atomwise_weight != 0

    target_atomwise = None
    target_grad = None
    target_stress = None

    dataset = []
    for i in dat:
        info = {}
        if id_prop_csv_file:
            file_name = i[0]
            tmp = [float(j) for j in i[1:]]
            info["jid"] = file_name
            if len(tmp) == 1:
                tmp = tmp[0]
            else:
                multioutput = True
            info["target"] = tmp
            file_path = os.path.join(root_dir, file_name)
            if file_format == "poscar":
                atoms = Atoms.from_poscar(file_path)
            elif file_format == "cif":
                atoms = Atoms.from_cif(file_path)
            elif file_format == "xyz":
                atoms = Atoms.from_xyz(file_path, box_size=500)
            elif file_format == "pdb":
                atoms = Atoms.from_pdb(file_path, max_lat=500)
            else:
                raise NotImplementedError(
                    "File format not implemented", file_format
                )
            info["atoms"] = atoms.to_dict()
        else:
            info["target"] = i[target_key]
            info["atoms"] = i["atoms"]
            info["jid"] = i[id_key]

        if train_atom:
            target_atomwise = "atomwise_target"
            info["atomwise_target"] = i[atomwise_key]
        if train_grad:
            target_grad = "atomwise_grad"
            info["atomwise_grad"] = i[gradwise_key]
        if train_stress:
            stress = (
                voigt_6_to_full_3x3_stress(i[stresswise_key])
                if len(i[stresswise_key]) == 6
                else i[stresswise_key]
            )
            info["stresses"] = stress
            target_stress = "stresses"

        if "extra_features" in i:
            info["extra_features"] = i["extra_features"]
        dataset.append(info)

    print("len dataset", len(dataset))

    line_graph = config.model.alignn_layers > 0

    if multioutput:
        if not all(len(i) == len(n_outputs[0]) for i in n_outputs):
            raise ValueError("Make sure the outputs are of same size.")
        config.model.output_features = len(n_outputs[0])
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

    model = None
    # print('config.model',config.model)
    model = ALIGNNAtomWise(config.model)
    if restart_model_path is not None:
        print("Restarting the model training:", restart_model_path)

        rest_config = loadjson(
            restart_model_path.replace("current_model.pt", "config.json")
        )
        model_config = ALIGNNAtomWiseConfig(**rest_config["model"])
        model = ALIGNNAtomWise(model_config)
        model.load_state_dict(
            torch.load(restart_model_path, map_location=device)
        )
        model = model.to(device)

    (
        train_loader,
        val_loader,
        test_loader,
        prepare_batch,
    ) = get_train_val_loaders(
        dataset_array=dataset,
        target="target",
        target_atomwise=target_atomwise,
        target_grad=target_grad,
        target_stress=target_stress,
        n_train=config.n_train,
        n_val=config.n_val,
        n_test=config.n_test,
        train_ratio=config.train_ratio,
        val_ratio=config.val_ratio,
        test_ratio=config.test_ratio,
        line_graph=line_graph,
        batch_size=config.batch_size,
        atom_features=config.atom_features,
        neighbor_strategy=config.neighbor_strategy,
        standardize=config.atom_features != "cgcnn",
        id_tag=config.id_tag,
        pin_memory=config.pin_memory,
        workers=config.num_workers,
        save_dataloader=config.save_dataloader,
        use_canonize=config.use_canonize,
        filename=config.filename,
        cutoff=config.cutoff,
        cutoff_extra=config.cutoff_extra,
        max_neighbors=config.max_neighbors,
        output_features=config.model.output_features,
        classification_threshold=config.classification_threshold,
        target_multiplication_factor=config.target_multiplication_factor,
        standard_scalar_and_pca=config.standard_scalar_and_pca,
        keep_data_order=config.keep_data_order,
        output_dir=config.output_dir,
        use_lmdb=config.use_lmdb,
    )
    print("net parameters", sum(p.numel() for p in model.parameters()))
    optimizer = setup_optimizer(group_decay(model), config)

    t1 = time.time()
    print("rank", rank)
    print("world_size", world_size)
    trainer = Trainer(
        config,
        model,
        train_loader,
        val_loader,
        optimizer,
        rank,
        world_size,
        device,
    )
    trainer.train()
    t2 = time.time()
    print("Time taken (s)", t2 - t1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Atomistic Line Graph Neural Network"
    )
    parser.add_argument(
        "--root_dir",
        default="./",
        help="Folder with id_props.csv, structure files",
    )
    parser.add_argument(
        "--config_name",
        default="alignn/examples/sample_data/config_example.json",
        help="Name of the config file",
    )
    parser.add_argument(
        "--file_format",
        default="poscar",
        help="poscar/cif/xyz/pdb file format.",
    )
    parser.add_argument(
        "--classification_threshold",
        default=None,
        help="Threshold for converting into 0/1 class",
    )
    parser.add_argument(
        "--batch_size", default=None, help="Batch size, generally 64"
    )
    parser.add_argument(
        "--epochs", default=None, help="Number of epochs, generally 300"
    )
    parser.add_argument(
        "--target_key",
        default="total_energy",
        help="Name of the key for graph level data",
    )
    parser.add_argument(
        "--id_key", default="jid", help="Name of the key for graph level id"
    )
    parser.add_argument(
        "--force_key",
        default="forces",
        help="Name of key for gradient level data",
    )
    parser.add_argument(
        "--atomwise_key",
        default="forces",
        help="Name of key for atomwise level data",
    )
    parser.add_argument(
        "--stresswise_key",
        default="stresses",
        help="Name of the key for stress data",
    )
    parser.add_argument(
        "--output_dir", default="./", help="Folder to save outputs"
    )
    parser.add_argument(
        "--restart_model_path",
        default=None,
        help="Checkpoint file path for model",
    )
    parser.add_argument(
        "--device",
        default=None,
        help="set device for training the model [e.g. cpu, cuda, cuda:2]",
    )

    args = parser.parse_args(sys.argv[1:])
    world_size = int(torch.cuda.device_count())
    # print("world_size", world_size)

    if world_size > 1:
        torch.multiprocessing.spawn(
            train_for_folder,
            args=(
                world_size,
                args.root_dir,
                args.config_name,
                args.classification_threshold,
                args.batch_size,
                args.epochs,
                args.id_key,
                args.target_key,
                args.atomwise_key,
                args.force_key,
                args.stresswise_key,
                args.file_format,
                args.restart_model_path,
                args.output_dir,
            ),
            nprocs=world_size,
        )
    else:
        train_for_folder(
            0,
            world_size,
            args.root_dir,
            args.config_name,
            args.classification_threshold,
            args.batch_size,
            args.epochs,
            args.id_key,
            args.target_key,
            args.atomwise_key,
            args.force_key,
            args.stresswise_key,
            args.file_format,
            args.restart_model_path,
            args.output_dir,
        )
    try:
        cleanup(world_size)
    except Exception:
        pass
