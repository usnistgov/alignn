"""Module to prepare ALIGNN dataset."""

from pathlib import Path
from typing import Optional
import os
import torch
import dgl
import numpy as np
import pandas as pd
from jarvis.core.atoms import Atoms
from alignn.graphs import Graph, StructureDataset
from tqdm import tqdm

tqdm.pandas()


def load_graphs(
    dataset=[],
    name: str = "dft_3d",
    neighbor_strategy: str = "k-nearest",
    cutoff: float = 8,
    cutoff_extra: float = 3,
    max_neighbors: int = 12,
    cachedir: Optional[Path] = None,
    use_canonize: bool = False,
    id_tag="jid",
    # extra_feats_json=None,
    map_size=1e12,
):
    """Construct crystal graphs.

    Load only atomic number node features
    and bond displacement vector edge features.

    Resulting graphs have scheme e.g.
    ```
    Graph(num_nodes=12, num_edges=156,
          ndata_schemes={'atom_features': Scheme(shape=(1,)}
          edata_schemes={'r': Scheme(shape=(3,)})
    ```
    """

    def atoms_to_graph(atoms):
        """Convert structure dict to DGLGraph."""
        structure = (
            Atoms.from_dict(atoms) if isinstance(atoms, dict) else atoms
        )
        return Graph.atom_dgl_multigraph(
            structure,
            cutoff=cutoff,
            cutoff_extra=cutoff_extra,
            atom_features="atomic_number",
            max_neighbors=max_neighbors,
            compute_line_graph=False,
            use_canonize=use_canonize,
            neighbor_strategy=neighbor_strategy,
        )

    if cachedir is not None:
        cachefile = cachedir / f"{name}-{neighbor_strategy}.bin"
    else:
        cachefile = None

    if cachefile is not None and cachefile.is_file():
        graphs, labels = dgl.load_graphs(str(cachefile))
    else:
        # print('dataset',dataset,type(dataset))
        print("Converting to graphs!")
        graphs = []
        # columns=dataset.columns
        for ii, i in tqdm(dataset.iterrows(), total=len(dataset)):
            # print('iooooo',i)
            atoms = i["atoms"]
            structure = (
                Atoms.from_dict(atoms) if isinstance(atoms, dict) else atoms
            )
            g = Graph.atom_dgl_multigraph(
                structure,
                cutoff=cutoff,
                cutoff_extra=cutoff_extra,
                atom_features="atomic_number",
                max_neighbors=max_neighbors,
                compute_line_graph=False,
                use_canonize=use_canonize,
                neighbor_strategy=neighbor_strategy,
                id=i[id_tag],
            )
            # print ('ii',ii)
            if "extra_features" in i:
                natoms = len(atoms["elements"])
                # if "extra_features" in columns:
                g.ndata["extra_features"] = torch.tensor(
                    [i["extra_features"] for n in range(natoms)]
                ).type(torch.get_default_dtype())
            graphs.append(g)

        # df = pd.DataFrame(dataset)
        # print ('df',df)

        # graphs = df["atoms"].progress_apply(atoms_to_graph).values
        # print ('graphs',graphs,graphs[0])
        if cachefile is not None:
            dgl.save_graphs(str(cachefile), graphs.tolist())

    return graphs


def get_torch_dataset(
    dataset=[],
    id_tag="jid",
    target="",
    target_atomwise="",
    target_grad="",
    target_stress="",
    neighbor_strategy="",
    atom_features="",
    use_canonize="",
    name="",
    line_graph="",
    cutoff=8.0,
    cutoff_extra=3.0,
    max_neighbors=12,
    classification=False,
    output_dir=".",
    tmp_name="dataset",
    sampler=None,
):
    """Get Torch Dataset."""
    df = pd.DataFrame(dataset)
    # df['natoms']=df['atoms'].apply(lambda x: len(x['elements']))
    # print(" data df", df)
    vals = np.array([ii[target] for ii in dataset])  # df[target].values
    print("data range", np.max(vals), np.min(vals))
    f = open(os.path.join(output_dir, tmp_name + "_data_range"), "w")
    line = "Max=" + str(np.max(vals)) + "\n"
    f.write(line)
    line = "Min=" + str(np.min(vals)) + "\n"
    f.write(line)
    f.close()

    graphs = load_graphs(
        df,
        name=name,
        neighbor_strategy=neighbor_strategy,
        use_canonize=use_canonize,
        cutoff=cutoff,
        cutoff_extra=cutoff_extra,
        max_neighbors=max_neighbors,
        id_tag=id_tag,
    )
    data = StructureDataset(
        df,
        graphs,
        target=target,
        target_atomwise=target_atomwise,
        target_grad=target_grad,
        target_stress=target_stress,
        atom_features=atom_features,
        line_graph=line_graph,
        id_tag=id_tag,
        classification=classification,
        sampler=sampler,
    )
    return data
