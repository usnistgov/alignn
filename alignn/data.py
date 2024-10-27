"""ALIGNN data loaders and DGLGraph utilities."""

import random
from typing import Optional
from torch.utils.data.distributed import DistributedSampler
import os
import torch
import numpy as np
from jarvis.db.figshare import data as jdata
from tqdm import tqdm
import math
from jarvis.db.jsonutils import dumpjson
from dgl.dataloading import GraphDataLoader
import pickle as pk
from sklearn.preprocessing import StandardScaler

tqdm.pandas()


def load_dataset(
    name: str = "dft_3d",
    target=None,
    limit: Optional[int] = None,
    classification_threshold: Optional[float] = None,
):
    """Load jarvis data."""
    d = jdata(name)
    data = []
    for i in d:
        if i[target] != "na" and not math.isnan(i[target]):
            if classification_threshold is not None:
                if i[target] <= classification_threshold:
                    i[target] = 0
                elif i[target] > classification_threshold:
                    i[target] = 1
                else:
                    raise ValueError(
                        "Check classification data type.",
                        i[target],
                        type(i[target]),
                    )
            data.append(i)
    d = data
    if limit is not None:
        d = d[:limit]
    # d = pd.DataFrame(d)
    # d = d.replace("na", np.nan)
    return d


# np.mean(mean_absolute_deviation(x,axis=0))
def mean_absolute_deviation(data, axis=None):
    """Get Mean absolute deviation."""
    return np.mean(np.absolute(data - np.mean(data, axis)), axis)


def get_id_train_val_test(
    total_size=1000,
    split_seed=123,
    train_ratio=None,
    val_ratio=0.1,
    test_ratio=0.1,
    n_train=None,
    n_test=None,
    n_val=None,
    keep_data_order=False,
):
    """Get train, val, test IDs."""
    if (
        train_ratio is None
        and val_ratio is not None
        and test_ratio is not None
    ):
        if train_ratio is None:
            assert val_ratio + test_ratio < 1
            train_ratio = 1 - val_ratio - test_ratio
            print("Using rest of the dataset except the test and val sets.")
        else:
            assert train_ratio + val_ratio + test_ratio <= 1
    # indices = list(range(total_size))
    if n_train is None:
        n_train = int(train_ratio * total_size)
    if n_test is None:
        n_test = int(test_ratio * total_size)
    if n_val is None:
        n_val = int(val_ratio * total_size)
    ids = list(np.arange(total_size))
    if not keep_data_order:
        random.seed(split_seed)
        random.shuffle(ids)
    # np.random.shuffle(ids)
    if n_train + n_val + n_test > total_size:
        raise ValueError(
            "Check total number of samples.",
            n_train + n_val + n_test,
            ">",
            total_size,
        )

    # shuffle consistently with https://github.com/txie-93/cgcnn/data.py
    # i.e. shuffle the index in place with standard library random.shuffle
    # first obtain only valid indices

    # test_size = round(N * 0.2)

    # full train/val test split
    # ids = ids[::-1]
    id_train = ids[:n_train]
    id_val = (
        ids[-(n_val + n_test) : -n_test]
        if n_test > 0
        else ids[-(n_val + n_test) :]
    )  # noqa:E203
    id_test = ids[-n_test:] if n_test > 0 else []
    return id_train, id_val, id_test


def get_train_val_loaders(
    dataset: str = "dft_3d",
    dataset_array=None,
    target: str = "formation_energy_peratom",
    target_atomwise: str = "",
    target_grad: str = "",
    target_stress: str = "",
    atom_features: str = "cgcnn",
    neighbor_strategy: str = "k-nearest",
    n_train=None,
    n_val=None,
    n_test=None,
    train_ratio=None,
    val_ratio=0.1,
    test_ratio=0.1,
    batch_size: int = 5,
    standardize: bool = False,
    line_graph: bool = True,
    split_seed: int = 123,
    workers: int = 0,
    pin_memory: bool = True,
    save_dataloader: bool = False,
    filename: str = "./",
    id_tag: str = "jid",
    use_canonize: bool = True,
    # use_ddp: bool = False,
    cutoff: float = 8.0,
    cutoff_extra: float = 3.0,
    max_neighbors: int = 12,
    classification_threshold: Optional[float] = None,
    target_multiplication_factor: Optional[float] = None,
    standard_scalar_and_pca=False,
    keep_data_order=False,
    output_features=1,
    output_dir=None,
    world_size=0,
    rank=0,
    use_lmdb: bool = True,
    dtype="float32",
):
    """Help function to set up JARVIS train and val dataloaders."""
    if use_lmdb:
        print("Using LMDB dataset.")
        from alignn.lmdb_dataset import get_torch_dataset
    else:
        print("Not using LMDB dataset, memory footprint maybe high.")
        from alignn.dataset import get_torch_dataset
    train_sample = filename + "_train.data"
    val_sample = filename + "_val.data"
    test_sample = filename + "_test.data"
    if os.path.exists(train_sample):
        print("If you are training from scratch, run")
        cmd = "rm -r " + train_sample + " " + val_sample + " " + test_sample
        print(cmd)
    # print ('output_dir data',output_dir)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    if (
        os.path.exists(train_sample)
        and os.path.exists(val_sample)
        and os.path.exists(test_sample)
        and save_dataloader
    ):
        print("Loading from saved file...")
        print("Make sure all the DataLoader params are same.")
        print("This module is made for debugging only.")
        train_loader = torch.load(train_sample)
        val_loader = torch.load(val_sample)
        test_loader = torch.load(test_sample)
        if train_loader.pin_memory != pin_memory:
            train_loader.pin_memory = pin_memory
        if test_loader.pin_memory != pin_memory:
            test_loader.pin_memory = pin_memory
        if val_loader.pin_memory != pin_memory:
            val_loader.pin_memory = pin_memory
        if train_loader.num_workers != workers:
            train_loader.num_workers = workers
        if test_loader.num_workers != workers:
            test_loader.num_workers = workers
        if val_loader.num_workers != workers:
            val_loader.num_workers = workers
        # print("train", len(train_loader.dataset))
        # print("val", len(val_loader.dataset))
        # print("test", len(test_loader.dataset))
    else:
        d = jdata(dataset) if dataset_array is None else dataset_array

        # for ii, i in enumerate(pc_y):
        #    d[ii][target] = pc_y[ii].tolist()

        dat = []
        if classification_threshold is not None:
            print(
                "Using ",
                classification_threshold,
                " for classifying ",
                target,
                " data.",
            )
            print("Converting target data into 1 and 0.")
        all_targets = []

        # TODO:make an all key in qm9_dgl
        if dataset == "qm9_dgl" and target == "all":
            print("Making all qm9_dgl")
            tmp = []
            for ii in d:
                ii["all"] = [
                    ii["mu"],
                    ii["alpha"],
                    ii["homo"],
                    ii["lumo"],
                    ii["gap"],
                    ii["r2"],
                    ii["zpve"],
                    ii["U0"],
                    ii["U"],
                    ii["H"],
                    ii["G"],
                    ii["Cv"],
                ]
                tmp.append(ii)
            print("Made all qm9_dgl")
            d = tmp
        for i in d:
            if isinstance(i[target], list):  # multioutput target
                all_targets.append(torch.tensor(i[target]))
                dat.append(i)

            elif (
                i[target] is not None
                and i[target] != "na"
                and not math.isnan(i[target])
            ):
                if target_multiplication_factor is not None:
                    i[target] = i[target] * target_multiplication_factor
                if classification_threshold is not None:
                    if i[target] <= classification_threshold:
                        i[target] = 0
                    elif i[target] > classification_threshold:
                        i[target] = 1
                    else:
                        raise ValueError(
                            "Check classification data type.",
                            i[target],
                            type(i[target]),
                        )
                dat.append(i)
                all_targets.append(i[target])

        # id_test = ids[-test_size:]
        # if standardize:
        #    data.setup_standardizer(id_train)
        id_train, id_val, id_test = get_id_train_val_test(
            total_size=len(dat),
            split_seed=split_seed,
            train_ratio=train_ratio,
            val_ratio=val_ratio,
            test_ratio=test_ratio,
            n_train=n_train,
            n_test=n_test,
            n_val=n_val,
            keep_data_order=keep_data_order,
        )
        ids_train_val_test = {}
        ids_train_val_test["id_train"] = [dat[i][id_tag] for i in id_train]
        ids_train_val_test["id_val"] = [dat[i][id_tag] for i in id_val]
        ids_train_val_test["id_test"] = [dat[i][id_tag] for i in id_test]
        dumpjson(
            data=ids_train_val_test,
            filename=os.path.join(output_dir, "ids_train_val_test.json"),
        )
        dataset_train = [dat[x] for x in id_train]
        dataset_val = [dat[x] for x in id_val]
        dataset_test = [dat[x] for x in id_test]

        if standard_scalar_and_pca:
            y_data = [i[target] for i in dataset_train]
            # pipe = Pipeline([('scale', StandardScaler())])
            if not isinstance(y_data[0], list):
                print("Running StandardScalar")
                y_data = np.array(y_data).reshape(-1, 1)
            sc = StandardScaler()

            sc.fit(y_data)
            print("Mean", sc.mean_)
            print("Variance", sc.var_)
            try:
                print("New max", max(y_data))
                print("New min", min(y_data))
            except Exception as exp:
                print(exp)
                pass
            # pc = PCA(n_components=output_features)
            # pipe = Pipeline(
            #    [
            #        ("scale", StandardScaler()),
            #        ("reduce_dims", PCA(n_components=output_features)),
            #    ]
            # )
            pk.dump(sc, open(os.path.join(output_dir, "sc.pkl"), "wb"))
            # pc = PCA(n_components=10)
            # pc.fit(y_data)
            # pk.dump(pc, open("pca.pkl", "wb"))

        if classification_threshold is None:
            try:
                from sklearn.metrics import mean_absolute_error

                print("MAX val:", max(all_targets))
                print("MIN val:", min(all_targets))
                print("MAD:", mean_absolute_deviation(all_targets))
                try:
                    f = open(os.path.join(output_dir, "mad"), "w")
                    line = "MAX val:" + str(max(all_targets)) + "\n"
                    line += "MIN val:" + str(min(all_targets)) + "\n"
                    line += (
                        "MAD val:"
                        + str(mean_absolute_deviation(all_targets))
                        + "\n"
                    )
                    f.write(line)
                    f.close()
                except Exception as exp:
                    print("Cannot write mad", exp)
                    pass
                # Random model precited value
                x_bar = np.mean(np.array([i[target] for i in dataset_train]))
                baseline_mae = mean_absolute_error(
                    np.array([i[target] for i in dataset_test]),
                    np.array([x_bar for i in dataset_test]),
                )
                print("Baseline MAE:", baseline_mae)
            except Exception as exp:
                print("Data error", exp)
                pass

        if world_size > 1:
            use_ddp = True
            train_sampler = DistributedSampler(
                dataset_train, num_replicas=world_size, rank=rank
            )
            val_sampler = DistributedSampler(
                dataset_val, num_replicas=world_size, rank=rank
            )
        else:
            use_ddp = False
            train_sampler = None
            val_sampler = None
        tmp_name = filename + "train_data"
        train_data = get_torch_dataset(
            dataset=dataset_train,
            id_tag=id_tag,
            atom_features=atom_features,
            target=target,
            target_atomwise=target_atomwise,
            target_grad=target_grad,
            target_stress=target_stress,
            neighbor_strategy=neighbor_strategy,
            use_canonize=use_canonize,
            name=dataset,
            line_graph=line_graph,
            cutoff=cutoff,
            cutoff_extra=cutoff_extra,
            max_neighbors=max_neighbors,
            classification=classification_threshold is not None,
            output_dir=output_dir,
            sampler=train_sampler,
            tmp_name=tmp_name,
            dtype=dtype,
            # tmp_name="train_data",
        )
        tmp_name = filename + "val_data"
        val_data = (
            get_torch_dataset(
                dataset=dataset_val,
                id_tag=id_tag,
                atom_features=atom_features,
                target=target,
                target_atomwise=target_atomwise,
                target_grad=target_grad,
                target_stress=target_stress,
                neighbor_strategy=neighbor_strategy,
                use_canonize=use_canonize,
                name=dataset,
                line_graph=line_graph,
                cutoff=cutoff,
                cutoff_extra=cutoff_extra,
                sampler=val_sampler,
                max_neighbors=max_neighbors,
                classification=classification_threshold is not None,
                output_dir=output_dir,
                tmp_name=tmp_name,
                dtype=dtype,
                # tmp_name="val_data",
            )
            if len(dataset_val) > 0
            else None
        )
        tmp_name = filename + "test_data"
        test_data = (
            get_torch_dataset(
                dataset=dataset_test,
                id_tag=id_tag,
                atom_features=atom_features,
                target=target,
                target_atomwise=target_atomwise,
                target_grad=target_grad,
                target_stress=target_stress,
                neighbor_strategy=neighbor_strategy,
                use_canonize=use_canonize,
                name=dataset,
                line_graph=line_graph,
                cutoff=cutoff,
                cutoff_extra=cutoff_extra,
                max_neighbors=max_neighbors,
                classification=classification_threshold is not None,
                output_dir=output_dir,
                tmp_name=tmp_name,
                dtype=dtype,
                # tmp_name="test_data",
            )
            if len(dataset_test) > 0
            else None
        )

        collate_fn = train_data.collate
        # print("line_graph,line_dih_graph", line_graph, line_dih_graph)
        if line_graph:
            collate_fn = train_data.collate_line_graph

        # use a regular pytorch dataloader
        train_loader = GraphDataLoader(
            # train_loader = DataLoader(
            train_data,
            batch_size=batch_size,
            shuffle=True,
            collate_fn=collate_fn,
            drop_last=True,
            num_workers=workers,
            pin_memory=pin_memory,
            use_ddp=use_ddp,
        )

        val_loader = GraphDataLoader(
            # val_loader = DataLoader(
            val_data,
            batch_size=batch_size,
            shuffle=False,
            collate_fn=collate_fn,
            drop_last=True,
            num_workers=workers,
            pin_memory=pin_memory,
            use_ddp=use_ddp,
        )

        test_loader = (
            GraphDataLoader(
                # DataLoader(
                test_data,
                batch_size=1,
                shuffle=False,
                collate_fn=collate_fn,
                drop_last=False,
                num_workers=workers,
                pin_memory=pin_memory,
                use_ddp=use_ddp,
            )
            if len(dataset_test) > 0
            else None
        )

        if save_dataloader:
            torch.save(train_loader, train_sample)
            if val_loader is not None:
                torch.save(val_loader, val_sample)
            if test_loader is not None:
                torch.save(test_loader, test_sample)

    print("n_train:", len(train_loader.dataset))
    print("n_val  :", len(val_loader.dataset) if val_loader is not None else 0)
    print(
        "n_test :", len(test_loader.dataset) if test_loader is not None else 0
    )
    return (
        train_loader,
        val_loader,
        test_loader,
        train_loader.dataset.prepare_batch,
    )
