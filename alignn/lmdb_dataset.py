"""Module to prepare LMDB ALIGNN dataset."""

import os
import numpy as np
import lmdb
from jarvis.core.atoms import Atoms
from jarvis.db.figshare import data
from alignn.graphs import Graph
import pickle as pk
from torch.utils.data import Dataset
import torch
from tqdm import tqdm
from typing import List, Tuple
import dgl


def prepare_line_graph_batch(
    batch: Tuple[Tuple[dgl.DGLGraph, dgl.DGLGraph], torch.Tensor],
    device=None,
    non_blocking=False,
):
    """Send line graph batch to device.

    Note: the batch is a nested tuple, with the graph and line graph together
    """
    g, lg, t, id = batch
    batch = (
        (
            g.to(device, non_blocking=non_blocking),
            lg.to(device, non_blocking=non_blocking),
        ),
        t.to(device, non_blocking=non_blocking),
    )

    return batch


class TorchLMDBDataset(Dataset):
    """Dataset of crystal DGLGraphs using LMDB."""

    def __init__(self, lmdb_path="", line_graph=True, ids=[]):
        """Intitialize with path and ids array."""
        super(TorchLMDBDataset, self).__init__()
        self.lmdb_path = lmdb_path
        self.ids = ids
        self.line_graph = line_graph
        self.env = lmdb.open(
            self.lmdb_path, readonly=True, lock=False, readahead=False
        )
        with self.env.begin() as txn:
            self.length = txn.stat()["entries"]
        self.prepare_batch = prepare_line_graph_batch

    def __len__(self):
        """Get length."""
        return self.length

    def __getitem__(self, idx):
        """Get sample."""
        if self.env is None:
            self.env = lmdb.open(
                self.lmdb_path, readonly=True, lock=False, readahead=False
            )

        with self.env.begin() as txn:
            serialized_data = txn.get(f"{idx}".encode())
        if self.line_graph:
            graph, line_graph, lattice, label = pk.loads(serialized_data)
            return graph, line_graph, lattice, label
        else:
            graph, lattice, label = pk.loads(serialized_data)
            return graph, lattice, label

    def __getitem__(self, idx):
        """Get sample."""
        try:
            if self.env is None:
                self.env = lmdb.open(
                    self.lmdb_path, readonly=True, lock=False, readahead=False
                )

            with self.env.begin() as txn:
                serialized_data = txn.get(f"{idx}".encode())
        except lmdb.MapResizedError:
            # Close and reopen the environment
            self.close()
            self.env = lmdb.open(
                self.lmdb_path, readonly=True, lock=False, readahead=False
            )
            with self.env.begin() as txn:
                serialized_data = txn.get(f"{idx}".encode())

        if self.line_graph:
            graph, line_graph, lattice, label = pk.loads(serialized_data)
            return graph, line_graph, lattice, label
        else:
            graph, lattice, label = pk.loads(serialized_data)
            return graph, lattice, label

    def close(self):
        """Close connection."""
        self.env.close()

    def __del__(self):
        """Delete connection."""
        self.close()

    @staticmethod
    def collate(samples: List[Tuple[dgl.DGLGraph, torch.Tensor]]):
        """Dataloader helper to batch graphs cross `samples`."""
        # print('samples',samples)
        graphs, lattices, labels = map(list, zip(*samples))
        # graphs, lgs, labels = map(list, zip(*samples))
        batched_graph = dgl.batch(graphs)
        if len(labels[0].size()) > 0:
            return batched_graph, torch.stack(lattices), torch.stack(labels)
        else:
            return batched_graph, torch.stack(lattices), torch.tensor(labels)

    @staticmethod
    def collate_line_graph(
        samples: List[Tuple[dgl.DGLGraph, dgl.DGLGraph, torch.Tensor]],
    ):
        """Dataloader helper to batch graphs cross `samples`."""
        graphs, line_graphs, lattices, labels = map(list, zip(*samples))
        batched_graph = dgl.batch(graphs)
        batched_line_graph = dgl.batch(line_graphs)
        if len(labels[0].size()) > 0:
            return (
                batched_graph,
                batched_line_graph,
                torch.stack(lattices),
                torch.stack(labels),
            )
        else:
            return (
                batched_graph,
                batched_line_graph,
                torch.stack(lattices),
                torch.tensor(labels),
            )


def get_torch_dataset(
    dataset=[],
    id_tag="jid",
    target="",
    target_atomwise="",
    target_grad="",
    target_stress="",
    target_additional_output="",
    neighbor_strategy="k-nearest",
    atom_features="cgcnn",
    use_canonize="",
    name="",
    line_graph=True,
    cutoff=8.0,
    cutoff_extra=3.0,
    max_neighbors=12,
    classification=False,
    sampler=None,
    output_dir=".",
    tmp_name="dataset",
    map_size=1e12,
    read_existing=True,
    dtype="float32",
    rank=0,
    world_size=0,
):
    """Get Torch Dataset with LMDB."""
    # read_existing=False
    vals = np.array([ii[target] for ii in dataset])  # df[target].values
    print("data range", np.max(vals), np.min(vals))
    print("line_graph", line_graph)
    f = open(os.path.join(output_dir, tmp_name + "_data_range"), "w")
    line = "Max=" + str(np.max(vals)) + "\n"
    f.write(line)
    line = "Min=" + str(np.min(vals)) + "\n"
    f.write(line)
    f.close()
    ids = []
    lock_path = os.path.join(tmp_name + "_done.lock")

    ids_path = os.path.join(tmp_name, "ids.txt")
    if rank == 0 and (
        not os.path.exists(tmp_name) or not os.path.exists(lock_path)
    ):
        print(f"[RANK {rank}] Creating LMDB...")
        os.makedirs(tmp_name, exist_ok=True)
        env = lmdb.open(tmp_name, map_size=int(map_size))

        ids = []
        with env.begin(write=True) as txn:
            for idx, (d) in tqdm(enumerate(dataset), total=len(dataset)):
                ids.append(d[id_tag])
                # g, lg = Graph.atom_dgl_multigraph(
                atoms = Atoms.from_dict(d["atoms"])
                g = Graph.atom_dgl_multigraph(
                    atoms,
                    cutoff=float(cutoff),
                    max_neighbors=max_neighbors,
                    atom_features=atom_features,
                    compute_line_graph=line_graph,
                    use_canonize=use_canonize,
                    cutoff_extra=cutoff_extra,
                    neighbor_strategy=neighbor_strategy,
                    dtype=dtype,
                )
                if line_graph:
                    g, lg = g
                lattice = torch.tensor(atoms.lattice_mat).type(
                    torch.get_default_dtype()
                )
                label = torch.tensor(d[target]).type(torch.get_default_dtype())
                natoms = len(d["atoms"]["elements"])
                # print('label',label,label.view(-1).long())
                if classification:
                    label = label.long()
                    # label = label.view(-1).long()
                if "extra_features" in d:
                    g.ndata["extra_features"] = torch.tensor(
                        [d["extra_features"] for n in range(natoms)]
                    ).type(torch.get_default_dtype())
                if target_atomwise is not None and target_atomwise != "":
                    g.ndata[target_atomwise] = torch.tensor(
                        np.array(d[target_atomwise])
                    ).type(torch.get_default_dtype())
                if target_grad is not None and target_grad != "":
                    # print('grad', np.array(d[target_grad]))
                    # print('grad shape',np.array(d[target_grad]).shape)
                    arr = np.array(d[target_grad])
                    try:
                        g.ndata[target_grad] = torch.tensor(arr).type(
                            torch.get_default_dtype()
                        )
                    except Exception:
                        arr = arr.reshape(1, -1)
                        g.ndata[target_grad] = torch.tensor(arr).type(
                            torch.get_default_dtype()
                        )
                        # print('arr',arr.shape)
                if target_stress is not None and target_stress != "":
                    stress = np.array(d[target_stress])
                    g.ndata[target_stress] = torch.tensor(
                        np.array([stress for ii in range(g.number_of_nodes())])
                    ).type(torch.get_default_dtype())
                if (
                    target_additional_output is not None
                    and target_additional_output != ""
                ):
                    additional_output = np.array(d[target_additional_output])
                    g.ndata[target_additional_output] = torch.tensor(
                        ([additional_output for ii in range(natoms)])
                    ).type(torch.get_default_dtype())

                # labels.append(label)
                if line_graph:
                    serialized_data = pk.dumps((g, lg, lattice, label))
                else:
                    serialized_data = pk.dumps((g, lattice, label))
                txn.put(f"{idx}".encode(), serialized_data)

        env.close()
        with open(ids_path, "w") as f:
            for i in ids:
                f.write(i + "\n")
        # with open(lock_path, "w") as f:
        #    f.write("done")

    if world_size > 1:
        if rank != 0:
            while not os.path.exists(lock_path):
                print(f"[RANK {rank}] Waiting for dataset...")
                time.sleep(1)  # wait for rank 0 to finish
        torch.distributed.barrier()  # synchronizes all ranks

    with open(os.path.join(tmp_name, "ids.txt"), "r") as f:
        ids = [line.strip() for line in f]

    return TorchLMDBDataset(lmdb_path=tmp_name, line_graph=line_graph, ids=ids)


if __name__ == "__main__":
    dataset = data("dft_2d")
    lmdb_dataset = get_torch_dataset(
        dataset=dataset, target="formation_energy_peratom"
    )
    print(lmdb_dataset)
