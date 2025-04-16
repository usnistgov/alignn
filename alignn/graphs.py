"""Module to generate networkx graphs."""

from jarvis.core.atoms import get_supercell_dims
from jarvis.core.specie import Specie
from jarvis.core.utils import random_colors
import numpy as np
import pandas as pd
from collections import OrderedDict
from jarvis.analysis.structure.neighbors import NeighborsAnalysis
from jarvis.core.specie import chem_data, get_node_attributes
import math
from collections import defaultdict
from typing import List, Tuple, Sequence, Optional
from dgl.data import DGLDataset
import torch
import dgl
from tqdm import tqdm
from jarvis.core.atoms import Atoms
from alignn.models.utils import (
    compute_cartesian_coordinates,
    #    compute_pair_vector_and_distance,
)

# import matgl


def temp_graph(
    atoms=None, cutoff=4.0, atom_features="atomic_number", dtype="float32"
):
    """Construct a graph for a given cutoff."""
    TORCH_DTYPES = {
        "float16": torch.float16,
        "float32": torch.float32,
        "float64": torch.float64,
        "bfloat": torch.bfloat16,
    }
    dtype = TORCH_DTYPES[dtype]
    u, v, r, d, images, atom_feats = [], [], [], [], [], []
    elements = atoms.elements

    # Loop over each atom in the structure
    for ii, i in enumerate(atoms.cart_coords):
        # Get neighbors within the cutoff distance
        neighs = atoms.lattice.get_points_in_sphere(
            atoms.frac_coords, i, cutoff, distance_vector=True
        )

        # Filter out self-loops (exclude cases where atom is bonded to itself)
        valid_indices = neighs[2] != ii

        u.extend([ii] * np.sum(valid_indices))
        d.extend(neighs[1][valid_indices])
        v.extend(neighs[2][valid_indices])
        images.extend(neighs[3][valid_indices])
        r.extend(neighs[4][valid_indices])

        feat = list(
            get_node_attributes(elements[ii], atom_features=atom_features)
        )
        atom_feats.append(feat)

    # Create DGL graph
    g = dgl.graph((np.array(u), np.array(v)))
    atom_feats = np.array(atom_feats)
    # Add data to the graph with the specified dtype
    # print('atom_feats',atom_feats,atom_feats.shape)
    g.ndata["atom_features"] = torch.tensor(atom_feats, dtype=dtype)
    g.ndata["Z"] = torch.tensor(atom_feats, dtype=torch.int64)
    g.edata["r"] = torch.tensor(np.array(r), dtype=dtype)
    g.edata["d"] = torch.tensor(d, dtype=dtype)
    # g.edata["pbc_offset"] = torch.tensor(images, dtype=dtype)
    # g.edata["pbc_offshift"] = torch.tensor(images, dtype=dtype)
    g.edata["images"] = torch.tensor(images, dtype=dtype)
    # node_type = torch.tensor([0 for i in range(len(atoms.atomic_numbers))])
    # g.ndata["node_type"] = node_type
    # lattice_mat = atoms.lattice_mat
    # g.ndata["lattice"] = torch.tensor(
    #   [lattice_mat for ii in range(g.num_nodes())]
    # , dtype=dtype)
    # g.edata["lattice"] = torch.tensor(
    #   [lattice_mat for ii in range(g.num_edges())]
    # , dtype=dtype)
    g.ndata["pos"] = torch.tensor(atoms.cart_coords, dtype=dtype)
    g.ndata["frac_coords"] = torch.tensor(atoms.frac_coords, dtype=dtype)

    return g, u, v, r


def radius_graph_jarvis(
    atoms,
    cutoff_extra=0.5,
    cutoff=4.0,
    atom_features="atomic_number",
    line_graph=True,
    dtype="float32",
    max_attempts=10,
):
    """Construct radius graph with jarvis tools."""
    count = 0
    while count <= max_attempts:
        # try:
        # Attempt to create the graph
        count += 1
        g, u, v, r = temp_graph(
            atoms=atoms,
            cutoff=cutoff,
            atom_features=atom_features,
            dtype=dtype,
        )
        # Check if all atoms are included as nodes
        if g.num_nodes() == len(atoms.elements):
            # print(f"Graph constructed with cutoff: {cutoff}")
            break  # Exit the loop when successful
        # Increment the cutoff if the graph is incomplete
        cutoff += cutoff_extra
        # print(f"Increasing cutoff to: {cutoff}")
    # except Exception as exp:
    #    # Handle exceptions and try again
    #    print(f"Graph construction failed: {exp,cutoff}")
    #    cutoff += cutoff_extra  # Try with a larger cutoff
    if count >= max_attempts:
        raise ValueError("Failed after", max_attempts, atoms)
    # Optional: Create a line graph if requested
    if line_graph:
        lg = g.line_graph(shared=True)
        lg.apply_edges(compute_bond_cosines)
        return g, lg

    return g


def canonize_edge(
    src_id,
    dst_id,
    src_image,
    dst_image,
):
    """Compute canonical edge representation.

    Sort vertex ids
    shift periodic images so the first vertex is in (0,0,0) image
    """
    # store directed edges src_id <= dst_id
    if dst_id < src_id:
        src_id, dst_id = dst_id, src_id
        src_image, dst_image = dst_image, src_image

    # shift periodic images so that src is in (0,0,0) image
    if not np.array_equal(src_image, (0, 0, 0)):
        shift = src_image
        src_image = tuple(np.subtract(src_image, shift))
        dst_image = tuple(np.subtract(dst_image, shift))

    assert src_image == (0, 0, 0)

    return src_id, dst_id, src_image, dst_image


def nearest_neighbor_edges(
    atoms=None,
    cutoff=8,
    max_neighbors=12,
    id=None,
    use_canonize=False,
):
    """Construct k-NN edge list."""
    # returns List[List[Tuple[site, distance, index, image]]]
    all_neighbors = atoms.get_all_neighbors(r=cutoff)

    # if a site has too few neighbors, increase the cutoff radius
    min_nbrs = min(len(neighborlist) for neighborlist in all_neighbors)

    attempt = 0
    # print ('cutoff=',all_neighbors)
    if min_nbrs < max_neighbors:
        # print("extending cutoff radius!", attempt, cutoff, id)
        lat = atoms.lattice
        if cutoff < max(lat.a, lat.b, lat.c):
            r_cut = max(lat.a, lat.b, lat.c)
        else:
            r_cut = 2 * cutoff
        attempt += 1

        return nearest_neighbor_edges(
            atoms=atoms,
            use_canonize=use_canonize,
            cutoff=r_cut,
            max_neighbors=max_neighbors,
            id=id,
        )
    # build up edge list
    # NOTE: currently there's no guarantee that this creates undirected graphs
    # An undirected solution would build the full edge list where nodes are
    # keyed by (index, image), and ensure each edge has a complementary edge

    # indeed, JVASP-59628 is an example of a calculation where this produces
    # a graph where one site has no incident edges!

    # build an edge dictionary u -> v
    # so later we can run through the dictionary
    # and remove all pairs of edges
    # so what's left is the odd ones out
    edges = defaultdict(set)
    for site_idx, neighborlist in enumerate(all_neighbors):
        # sort on distance
        neighborlist = sorted(neighborlist, key=lambda x: x[2])
        distances = np.array([nbr[2] for nbr in neighborlist])
        ids = np.array([nbr[1] for nbr in neighborlist])
        images = np.array([nbr[3] for nbr in neighborlist])

        # find the distance to the k-th nearest neighbor
        max_dist = distances[max_neighbors - 1]
        # max_dist = distances[max_neighbors - 1]

        # keep all edges out to the neighbor shell of the k-th neighbor
        ids = ids[distances <= max_dist]
        images = images[distances <= max_dist]
        distances = distances[distances <= max_dist]

        # keep track of cell-resolved edges
        # to enforce undirected graph construction
        for dst, image in zip(ids, images):
            src_id, dst_id, src_image, dst_image = canonize_edge(
                site_idx, dst, (0, 0, 0), tuple(image)
            )
            if use_canonize:
                edges[(src_id, dst_id)].add(dst_image)
            else:
                edges[(site_idx, dst)].add(tuple(image))

    return edges, images


def build_undirected_edgedata(
    atoms=None,
    edges={},
):
    """Build undirected graph data from edge set.

    edges: dictionary mapping (src_id, dst_id) to set of dst_image
    r: cartesian displacement vector from src -> dst
    """
    # second pass: construct *undirected* graph
    # import pprint
    u, v, r, all_images = [], [], [], []
    for (src_id, dst_id), images in edges.items():
        for dst_image in images:
            # fractional coordinate for periodic image of dst
            dst_coord = atoms.frac_coords[dst_id] + dst_image
            # cartesian displacement vector pointing from src -> dst
            d = atoms.lattice.cart_coords(
                dst_coord - atoms.frac_coords[src_id]
            )
            # if np.linalg.norm(d)!=0:
            # print ('jv',dst_image,d)
            # add edges for both directions
            for uu, vv, dd in [(src_id, dst_id, d), (dst_id, src_id, -d)]:
                u.append(uu)
                v.append(vv)
                r.append(dd)
                all_images.append(dst_image)
    u, v, r = (np.array(x) for x in (u, v, r))
    u = torch.tensor(u)
    v = torch.tensor(v)
    r = torch.tensor(r).type(torch.get_default_dtype())
    all_images = torch.tensor(all_images).type(torch.get_default_dtype())

    return u, v, r, all_images


def radius_graph(
    atoms=None,
    cutoff=5,
    bond_tol=0.5,
    id=None,
    atol=1e-5,
    cutoff_extra=0.5,
):
    """Construct edge list for radius graph."""

    def temp_graph(cutoff=5):
        """Construct edge list for radius graph."""
        cart_coords = torch.tensor(atoms.cart_coords).type(
            torch.get_default_dtype()
        )
        frac_coords = torch.tensor(atoms.frac_coords).type(
            torch.get_default_dtype()
        )
        lattice_mat = torch.tensor(atoms.lattice_mat).type(
            torch.get_default_dtype()
        )
        # elements = atoms.elements
        X_src = cart_coords
        num_atoms = X_src.shape[0]
        # determine how many supercells are needed for the cutoff radius
        recp = 2 * math.pi * torch.linalg.inv(lattice_mat).T
        recp_len = torch.tensor(
            [i for i in (torch.sqrt(torch.sum(recp**2, dim=1)))]
        )
        maxr = torch.ceil((cutoff + bond_tol) * recp_len / (2 * math.pi))
        nmin = torch.floor(torch.min(frac_coords, dim=0)[0]) - maxr
        nmax = torch.ceil(torch.max(frac_coords, dim=0)[0]) + maxr
        # construct the supercell index list

        all_ranges = [
            torch.arange(x, y, dtype=torch.get_default_dtype())
            for x, y in zip(nmin, nmax)
        ]
        cell_images = torch.cartesian_prod(*all_ranges)

        # tile periodic images into X_dst
        # index id_dst into X_dst maps to atom id as id_dest % num_atoms
        X_dst = (cell_images @ lattice_mat)[:, None, :] + X_src
        # cell_images = cell_images[:,None,:]+cell_images
        # print('cell_images',cell_images,cell_images.shape)
        X_dst = X_dst.reshape(-1, 3)
        # pairwise distances between atoms in (0,0,0) cell
        # and atoms in all periodic image
        dist = torch.cdist(
            X_src, X_dst, compute_mode="donot_use_mm_for_euclid_dist"
        )
        # u, v = torch.nonzero(dist <= cutoff, as_tuple=True)
        # print("u1v1", u, v, u.shape, v.shape)
        neighbor_mask = torch.bitwise_and(
            dist <= cutoff,
            ~torch.isclose(
                dist,
                torch.tensor([0]).type(torch.get_default_dtype()),
                atol=atol,
            ),
        )

        # get node indices for edgelist from neighbor mask
        u, v = torch.where(neighbor_mask)
        # cell_images=cell_images[neighbor_mask]
        # u, v = torch.where(neighbor_mask)
        # print("u2v2", u, v, u.shape, v.shape)
        # print("v1", v, v.shape)
        # print("v2", v % num_atoms, (v % num_atoms).shape)
        cell_images = cell_images[v // num_atoms]

        r = (X_dst[v] - X_src[u]).float()
        # gk = dgl.knn_graph(X_dst, 12)
        # print("r", r, r.shape)
        # print("gk", gk)
        v = v % num_atoms
        g = dgl.graph((u, v))
        return g, u, v, r, cell_images

    # g, u, v, r, cell_images = temp_graph(cutoff)
    while True:  # (g.num_nodes()) != len(atoms.elements):
        # try:
        g, u, v, r, cell_images = temp_graph(cutoff)
        # g, u, v, r, cell_images = temp_graph(cutoff)
        # print(atoms)
        if (g.num_nodes()) == len(atoms.elements):
            return u, v, r, cell_images
        else:
            cutoff += cutoff_extra
            # print("cutoff", id, cutoff, atoms)

    # except Exception as exp:
    #    print("Graph exp", exp,atoms)
    #    cutoff += cutoff_extra
    #    pass
    # return u, v, r, cell_images

    return u, v, r, cell_images


###
def radius_graph_old(
    atoms=None,
    cutoff=5,
    bond_tol=0.5,
    id=None,
    atol=1e-5,
):
    """Construct edge list for radius graph."""
    cart_coords = torch.tensor(atoms.cart_coords).type(
        torch.get_default_dtype()
    )
    frac_coords = torch.tensor(atoms.frac_coords).type(
        torch.get_default_dtype()
    )
    lattice_mat = torch.tensor(atoms.lattice_mat).type(
        torch.get_default_dtype()
    )
    # elements = atoms.elements
    X_src = cart_coords
    num_atoms = X_src.shape[0]
    # determine how many supercells are needed for the cutoff radius
    recp = 2 * math.pi * torch.linalg.inv(lattice_mat).T
    recp_len = torch.tensor(
        [i for i in (torch.sqrt(torch.sum(recp**2, dim=1)))]
    )
    maxr = torch.ceil((cutoff + bond_tol) * recp_len / (2 * math.pi))
    nmin = torch.floor(torch.min(frac_coords, dim=0)[0]) - maxr
    nmax = torch.ceil(torch.max(frac_coords, dim=0)[0]) + maxr
    # construct the supercell index list

    all_ranges = [
        torch.arange(x, y, dtype=torch.get_default_dtype())
        for x, y in zip(nmin, nmax)
    ]
    cell_images = torch.cartesian_prod(*all_ranges)

    # tile periodic images into X_dst
    # index id_dst into X_dst maps to atom id as id_dest % num_atoms
    X_dst = (cell_images @ lattice_mat)[:, None, :] + X_src
    X_dst = X_dst.reshape(-1, 3)

    # pairwise distances between atoms in (0,0,0) cell
    # and atoms in all periodic image
    dist = torch.cdist(
        X_src, X_dst, compute_mode="donot_use_mm_for_euclid_dist"
    )
    # u, v = torch.nonzero(dist <= cutoff, as_tuple=True)
    # print("u1v1", u, v, u.shape, v.shape)
    neighbor_mask = torch.bitwise_and(
        dist <= cutoff,
        ~torch.isclose(
            dist, torch.tensor([0]).type(torch.get_default_dtype()), atol=atol
        ),
    )
    # get node indices for edgelist from neighbor mask
    u, v = torch.where(neighbor_mask)
    # print("u2v2", u, v, u.shape, v.shape)
    # print("v1", v, v.shape)
    # print("v2", v % num_atoms, (v % num_atoms).shape)

    r = (X_dst[v] - X_src[u]).float()
    # gk = dgl.knn_graph(X_dst, 12)
    # print("r", r, r.shape)
    # print("gk", gk)
    return u, v % num_atoms, r


###


def get_line_graph(
    g, lat=[], inner_cutoff=3.0, lighten_edges=False, backtracking=True
):
    """Generate a line graph object."""
    if not lighten_edges:
        lg = g.line_graph(shared=True, backtracking=backtracking)
        # lg.ndata["r"] = r
        lg.apply_edges(compute_bond_cosines)
        return lg
    else:
        # print(g)
        g.ndata["cart_coords"] = compute_cartesian_coordinates(g, lat)
        g.ndata["cart_coords"].requires_grad_(True)
        # r, bondlength = compute_pair_vector_and_distance(g)
        # dst_pos = g.ndata["cart_coords"][g.edges()[1]] + g.edata["images"]
        # src_pos = g.ndata["cart_coords"][g.edges()[0]]
        # bond_vec = dst_pos - src_pos
        # bond_dist = torch.norm(bond_vec, dim=1)
        # pos = g.ndata["cart_coords"]
        # g.edata["bond_dist"] = bond_dist
        # g.edata["r"] = bond_vec

        src, dst = g.edges()  # shape: [E], [E]
        pos = g.ndata["cart_coords"]
        src1 = src.unsqueeze(1)  # [E,1]
        dst1 = dst.unsqueeze(1)  # [E,1]
        src2 = src.unsqueeze(0)  # [1,E]
        dst2 = dst.unsqueeze(0)  # [1,E]
        # Broadcasted match on center node
        center_match = dst1 == src2  # [E, E] -> bool matrix
        # Get u, v, w for matching triples
        u = src1.expand(-1, len(src))  # [E, E]
        # v = dst1.expand(-1, len(src))  # [E, E]
        # v2 = src2.expand(len(src), -1)  # [E, E]
        w = dst2.expand(len(src), -1)  # [E, E]
        # Mask out u == w (no backtracking)
        non_backtrack = u != w
        # Compute distance from u to w for all pairs (eid1, eid2)
        pos_u = pos[u]
        pos_w = pos[w]
        uw_dist = torch.norm(pos_u - pos_w, dim=-1)  # [E, E]
        # Apply angular cutoff
        angle_mask = center_match & non_backtrack & (uw_dist < inner_cutoff)
        angle_mask = center_match & (uw_dist < inner_cutoff)
        # angle_mask =  (uw_dist < inner_cutoff)
        # Get edge pairs (eid1, eid2) for the line graph
        eid1, eid2 = angle_mask.nonzero(as_tuple=True)
        # Create the line graph
        lg = dgl.graph((eid1, eid2), num_nodes=len(src))
        lg.ndata["r"] = g.edata["r"]  # bond_vec
        lg.apply_edges(compute_bond_cosines)
        return lg


class Graph(object):
    """Generate a graph object."""

    def __init__(
        self,
        nodes=[],
        node_attributes=[],
        edges=[],
        edge_attributes=[],
        color_map=None,
        labels=None,
    ):
        """
        Initialize the graph object.

        Args:
            nodes: IDs of the graph nodes as integer array.

            node_attributes: node features as multi-dimensional array.

            edges: connectivity as a (u,v) pair where u is
                   the source index and v the destination ID.

            edge_attributes: attributes for each connectivity.
                             as simple as euclidean distances.
        """
        self.nodes = nodes
        self.node_attributes = node_attributes
        self.edges = edges
        self.edge_attributes = edge_attributes
        self.color_map = color_map
        self.labels = labels

    @staticmethod
    def atom_dgl_multigraph(
        atoms=None,
        neighbor_strategy="k-nearest",
        cutoff=8.0,
        max_neighbors=12,
        atom_features="cgcnn",
        max_attempts=3,
        id: Optional[str] = None,
        compute_line_graph: bool = True,
        use_canonize: bool = True,
        # use_canonize: bool = False,
        use_lattice_prop: bool = False,
        cutoff_extra=3.5,
        dtype="float32",
        backtracking=True,
        inner_cutoff=3.0,
        lighten_edges=False,
    ):
        """Obtain a DGLGraph for Atoms object."""
        # print('id',id)
        # print('stratgery', neighbor_strategy)
        if neighbor_strategy == "k-nearest":
            edges, images = nearest_neighbor_edges(
                atoms=atoms,
                cutoff=cutoff,
                max_neighbors=max_neighbors,
                id=id,
                use_canonize=use_canonize,
            )
            u, v, r, images = build_undirected_edgedata(atoms, edges)
        elif neighbor_strategy == "radius_graph":
            # print('HERE')
            # import sys
            # sys.exit()
            u, v, r, images = radius_graph(
                atoms, cutoff=cutoff, cutoff_extra=cutoff_extra
            )
        elif neighbor_strategy == "radius_graph_jarvis":
            g, lg = radius_graph_jarvis(
                atoms,
                cutoff=cutoff,
                atom_features=atom_features,
                line_graph=compute_line_graph,
                dtype=dtype,
            )
            return g, lg
        else:
            raise ValueError("Not implemented yet", neighbor_strategy)
        # elif neighbor_strategy == "voronoi":
        #    edges = voronoi_edges(structure)

        # u, v, r = build_undirected_edgedata(atoms, edges)

        # build up atom attribute tensor
        # comp = atoms.composition.to_dict()
        # comp_dict = {}
        # c_ind = 0
        # for ii, jj in comp.items():
        #    if ii not in comp_dict:
        #        comp_dict[ii] = c_ind
        #        c_ind += 1
        sps_features = []
        # node_types = []
        for ii, s in enumerate(atoms.elements):
            feat = list(get_node_attributes(s, atom_features=atom_features))
            # if include_prdf_angles:
            #    feat=feat+list(prdf[ii])+list(adf[ii])
            sps_features.append(feat)
            # node_types.append(comp_dict[s])
        sps_features = np.array(sps_features)
        node_features = torch.tensor(sps_features).type(
            torch.get_default_dtype()
        )
        # print("u", u)
        # print("v", v)
        g = dgl.graph((u, v))
        g.ndata["atom_features"] = node_features
        # g.ndata["node_type"] = torch.tensor(node_types, dtype=torch.int64)
        # node_type = torch.tensor([0 for i in range(len(atoms.atm_num))])
        # g.ndata["node_type"] = node_type
        # print('g.ndata["node_type"]',g.ndata["node_type"])
        g.edata["r"] = torch.tensor(np.array(r)).type(
            torch.get_default_dtype()
        )
        # images=torch.tensor(images).type(torch.get_default_dtype())
        # print('images',images.shape,r.shape)
        # print('type',torch.get_default_dtype())
        g.edata["images"] = torch.tensor(np.array(images)).type(
            torch.get_default_dtype()
        )
        vol = atoms.volume
        g.ndata["V"] = torch.tensor([vol for ii in range(atoms.num_atoms)])
        # g.ndata["coords"] = torch.tensor(atoms.cart_coords).type(
        #    torch.get_default_dtype()
        # )
        g.ndata["frac_coords"] = torch.tensor(atoms.frac_coords).type(
            torch.get_default_dtype()
        )
        if use_lattice_prop:
            lattice_prop = np.array(
                [atoms.lattice.lat_lengths(), atoms.lattice.lat_angles()]
            ).flatten()
            # print('lattice_prop',lattice_prop)
            g.ndata["extra_features"] = torch.tensor(
                [lattice_prop for ii in range(atoms.num_atoms)]
            ).type(torch.get_default_dtype())
        # print("g", g)
        # g.edata["V"] = torch.tensor(
        #    [vol for ii in range(g.num_edges())]
        # )
        # lattice_mat = atoms.lattice_mat
        # g.edata["lattice_mat"] = torch.tensor(
        #    [lattice_mat for ii in range(g.num_edges())]
        # )

        if compute_line_graph:
            # construct atomistic line graph
            # (nodes are bonds, edges are bond pairs)
            # and add bond angle cosines as edge features
            # print("lighten_edges",lighten_edges)
            if lighten_edges:
                lg = get_line_graph(
                    g,
                    lat=torch.tensor(atoms.lattice_mat).type(
                        torch.get_default_dtype()
                    ),
                    inner_cutoff=inner_cutoff,
                    lighten_edges=lighten_edges,
                    backtracking=backtracking,
                )
            else:
                lg = g.line_graph(shared=True, backtracking=backtracking)
                lg.apply_edges(compute_bond_cosines)
            return g, lg
        else:
            return g

    @staticmethod
    def from_atoms(
        atoms=None,
        get_prim=False,
        zero_diag=False,
        node_atomwise_angle_dist=False,
        node_atomwise_rdf=False,
        features="basic",
        enforce_c_size=10.0,
        max_n=100,
        max_cut=5.0,
        verbose=False,
        make_colormap=True,
    ):
        """
        Get Networkx graph. Requires Networkx installation.

        Args:
             atoms: jarvis.core.Atoms object.

             rcut: cut-off after which distance will be set to zero
                   in the adjacency matrix.

             features: Node features.
                       'atomic_number': graph with atomic numbers only.
                       'cfid': 438 chemical descriptors from CFID.
                       'cgcnn': hot encoded 92 features.
                       'basic':10 features
                       'atomic_fraction': graph with atomic fractions
                                         in 103 elements.
                       array: array with CFID chemical descriptor names.
                       See: jarvis/core/specie.py

             enforce_c_size: minimum size of the simulation cell in Angst.
        """
        if get_prim:
            atoms = atoms.get_primitive_atoms
        dim = get_supercell_dims(atoms=atoms, enforce_c_size=enforce_c_size)
        atoms = atoms.make_supercell(dim)

        adj = np.array(atoms.raw_distance_matrix.copy())

        # zero out edges with bond length greater than threshold
        adj[adj >= max_cut] = 0

        if zero_diag:
            np.fill_diagonal(adj, 0.0)
        nodes = np.arange(atoms.num_atoms)
        if features == "atomic_number":
            node_attributes = np.array(
                [[np.array(Specie(i).Z)] for i in atoms.elements],
                dtype="float",
            )
        if features == "atomic_fraction":
            node_attributes = []
            fracs = atoms.composition.atomic_fraction_array
            for i in fracs:
                node_attributes.append(np.array([float(i)]))
            node_attributes = np.array(node_attributes)

        elif features == "basic":
            feats = [
                "Z",
                "coulmn",
                "row",
                "X",
                "atom_rad",
                "nsvalence",
                "npvalence",
                "ndvalence",
                "nfvalence",
                "first_ion_en",
                "elec_aff",
            ]
            node_attributes = []
            for i in atoms.elements:
                tmp = []
                for j in feats:
                    tmp.append(Specie(i).element_property(j))
                node_attributes.append(tmp)
            node_attributes = np.array(node_attributes, dtype="float")
        elif features == "cfid":
            node_attributes = np.array(
                [np.array(Specie(i).get_descrp_arr) for i in atoms.elements],
                dtype="float",
            )
        elif isinstance(features, list):
            node_attributes = []
            for i in atoms.elements:
                tmp = []
                for j in features:
                    tmp.append(Specie(i).element_property(j))
                node_attributes.append(tmp)
            node_attributes = np.array(node_attributes, dtype="float")
        else:
            print("Please check the input options.")
        if node_atomwise_rdf or node_atomwise_angle_dist:
            nbr = NeighborsAnalysis(
                atoms, max_n=max_n, verbose=verbose, max_cut=max_cut
            )
        if node_atomwise_rdf:
            node_attributes = np.concatenate(
                (node_attributes, nbr.atomwise_radial_dist()), axis=1
            )
            node_attributes = np.array(node_attributes, dtype="float")
        if node_atomwise_angle_dist:
            node_attributes = np.concatenate(
                (node_attributes, nbr.atomwise_angle_dist()), axis=1
            )
            node_attributes = np.array(node_attributes, dtype="float")

        # construct edge list
        uv = []
        edge_features = []
        for ii, i in enumerate(atoms.elements):
            for jj, j in enumerate(atoms.elements):
                bondlength = adj[ii, jj]
                if bondlength > 0:
                    uv.append((ii, jj))
                    edge_features.append(bondlength)

        edge_attributes = edge_features

        if make_colormap:
            sps = atoms.uniq_species
            color_dict = random_colors(number_of_colors=len(sps))
            new_colors = {}
            for i, j in color_dict.items():
                new_colors[sps[i]] = j
            color_map = []
            for ii, i in enumerate(atoms.elements):
                color_map.append(new_colors[i])
        return Graph(
            nodes=nodes,
            edges=uv,
            node_attributes=np.array(node_attributes),
            edge_attributes=np.array(edge_attributes),
            color_map=color_map,
        )

    def to_networkx(self):
        """Get networkx representation."""
        import networkx as nx

        graph = nx.DiGraph()
        graph.add_nodes_from(self.nodes)
        graph.add_edges_from(self.edges)
        for i, j in zip(self.edges, self.edge_attributes):
            graph.add_edge(i[0], i[1], weight=j)
        return graph

    @property
    def num_nodes(self):
        """Return number of nodes in the graph."""
        return len(self.nodes)

    @property
    def num_edges(self):
        """Return number of edges in the graph."""
        return len(self.edges)

    @classmethod
    def from_dict(self, d={}):
        """Constuct class from a dictionary."""
        return Graph(
            nodes=d["nodes"],
            edges=d["edges"],
            node_attributes=d["node_attributes"],
            edge_attributes=d["edge_attributes"],
            color_map=d["color_map"],
            labels=d["labels"],
        )

    def to_dict(self):
        """Provide dictionary representation of the Graph object."""
        info = OrderedDict()
        info["nodes"] = np.array(self.nodes).tolist()
        info["edges"] = np.array(self.edges).tolist()
        info["node_attributes"] = np.array(self.node_attributes).tolist()
        info["edge_attributes"] = np.array(self.edge_attributes).tolist()
        info["color_map"] = np.array(self.color_map).tolist()
        info["labels"] = np.array(self.labels).tolist()
        return info

    def __repr__(self):
        """Provide representation during print statements."""
        return "Graph({})".format(self.to_dict())

    @property
    def adjacency_matrix(self):
        """Provide adjacency_matrix of graph."""
        A = np.zeros((self.num_nodes, self.num_nodes))
        for edge, a in zip(self.edges, self.edge_attributes):
            A[edge] = a
        return A


class Standardize(torch.nn.Module):
    """Standardize atom_features: subtract mean and divide by std."""

    def __init__(self, mean: torch.Tensor, std: torch.Tensor):
        """Register featurewise mean and standard deviation."""
        super().__init__()
        self.mean = mean
        self.std = std

    def forward(self, g: dgl.DGLGraph):
        """Apply standardization to atom_features."""
        g = g.local_var()
        h = g.ndata.pop("atom_features")
        g.ndata["atom_features"] = (h - self.mean) / self.std
        return g


def prepare_dgl_batch(
    batch: Tuple[dgl.DGLGraph, torch.Tensor], device=None, non_blocking=False
):
    """Send batched dgl crystal graph to device."""
    g, t = batch
    batch = (
        g.to(device, non_blocking=non_blocking),
        t.to(device, non_blocking=non_blocking),
    )

    return batch


def prepare_line_graph_batch(
    batch: Tuple[Tuple[dgl.DGLGraph, dgl.DGLGraph], torch.Tensor],
    device=None,
    non_blocking=False,
):
    """Send line graph batch to device.

    Note: the batch is a nested tuple, with the graph and line graph together
    """
    g, lg, t = batch
    batch = (
        (
            g.to(device, non_blocking=non_blocking),
            lg.to(device, non_blocking=non_blocking),
        ),
        t.to(device, non_blocking=non_blocking),
    )

    return batch


# def prepare_batch(batch, device=None):
#     """Send tuple to device, including DGLGraphs."""
#     return tuple(x.to(device) for x in batch)


def compute_bond_cosines(edges):
    """Compute bond angle cosines from bond displacement vectors."""
    # line graph edge: (a, b), (b, c)
    # `a -> b -> c`
    # use law of cosines to compute angles cosines
    # negate src bond so displacements are like `a <- b -> c`
    # cos(theta) = ba \dot bc / (||ba|| ||bc||)
    r1 = -edges.src["r"]
    r2 = edges.dst["r"]
    bond_cosine = torch.sum(r1 * r2, dim=1) / (
        torch.norm(r1, dim=1) * torch.norm(r2, dim=1)
    )
    bond_cosine = torch.clamp(bond_cosine, -1, 1)
    # bond_cosine = torch.arccos((torch.clamp(bond_cosine, -1, 1)))
    # print (r1,r1.shape)
    # print (r2,r2.shape)
    # print (bond_cosine,bond_cosine.shape)
    return {"h": bond_cosine}


class StructureDataset(DGLDataset):
    """Dataset of crystal DGLGraphs."""

    def __init__(
        self,
        df: pd.DataFrame,
        graphs: Sequence[dgl.DGLGraph],
        target: str,
        target_atomwise="",
        target_grad="",
        target_stress="",
        atom_features="atomic_number",
        transform=None,
        line_graph=False,
        classification=False,
        id_tag="jid",
        sampler=None,
        lattices=None,
        dtype="float32",
    ):
        """Pytorch Dataset for atomistic graphs.

        `df`: pandas dataframe from e.g. jarvis.db.figshare.data
        `graphs`: DGLGraph representations corresponding to rows in `df`
        `target`: key for label column in `df`
        `target_grad`: For fitting forces etc.
        `target_atomwise`: For fitting bader charge on atoms etc.
        """
        self.df = df
        self.graphs = graphs
        self.target = target
        self.target_atomwise = target_atomwise
        self.target_grad = target_grad
        self.target_stress = target_stress
        self.line_graph = line_graph
        print("df", df)
        self.lattices = lattices
        self.labels = self.df[target]

        if (
            self.target_atomwise is not None and self.target_atomwise != ""
        ):  # and "" not in self.target_atomwise:
            # self.labels_atomwise = df[self.target_atomwise]
            self.labels_atomwise = []
            for ii, i in df.iterrows():
                self.labels_atomwise.append(
                    torch.tensor(np.array(i[self.target_atomwise])).type(
                        torch.get_default_dtype()
                    )
                )

        if (
            self.target_grad is not None and self.target_grad != ""
        ):  # and "" not in  self.target_grad :
            # self.labels_atomwise = df[self.target_atomwise]
            self.labels_grad = []
            for ii, i in df.iterrows():
                self.labels_grad.append(
                    torch.tensor(np.array(i[self.target_grad])).type(
                        torch.get_default_dtype()
                    )
                )
            # print (self.labels_atomwise)
        if (
            self.target_stress is not None and self.target_stress != ""
        ):  # and "" not in  self.target_stress :
            # self.labels_atomwise = df[self.target_atomwise]
            self.labels_stress = []
            for ii, i in df.iterrows():
                self.labels_stress.append(i[self.target_stress])
                # self.labels_stress.append(
                #    torch.tensor(np.array(i[self.target_stress])).type(
                #        torch.get_default_dtype()
                #    )
                # )
            # self.labels_stress = self.df[self.target_stress]

        self.ids = self.df[id_tag]
        self.labels = torch.tensor(self.df[target]).type(
            torch.get_default_dtype()
        )
        self.lattices = []
        for ii, i in df.iterrows():
            self.lattices.append(Atoms.from_dict(i["atoms"]).lattice_mat)

        self.lattices = torch.tensor(self.lattices).type(
            torch.get_default_dtype()
        )
        self.transform = transform

        features = self._get_attribute_lookup(atom_features)

        # load selected node representation
        # assume graphs contain atomic number in g.ndata["atom_features"]
        for i, g in enumerate(graphs):
            z = g.ndata.pop("atom_features")
            g.ndata["atomic_number"] = z
            z = z.type(torch.IntTensor).squeeze()
            f = torch.tensor(features[z]).type(torch.FloatTensor)
            if g.num_nodes() == 1:
                f = f.unsqueeze(0)
            g.ndata["atom_features"] = f
            if (
                self.target_atomwise is not None and self.target_atomwise != ""
            ):  # and "" not in self.target_atomwise:
                g.ndata[self.target_atomwise] = self.labels_atomwise[i]
            if (
                self.target_grad is not None and self.target_grad != ""
            ):  # and "" not in  self.target_grad:
                g.ndata[self.target_grad] = self.labels_grad[i]
            if (
                self.target_stress is not None and self.target_stress != ""
            ):  # and "" not in  self.target_stress:
                # print(
                #    "self.labels_stress[i]",
                #    [self.labels_stress[i] for ii in range(len(z))],
                # )
                g.ndata[self.target_stress] = torch.tensor(
                    [self.labels_stress[i] for ii in range(len(z))]
                ).type(torch.get_default_dtype())

        self.prepare_batch = prepare_dgl_batch
        if line_graph:
            self.prepare_batch = prepare_line_graph_batch

            print("building line graphs")
            self.line_graphs = []
            for g in tqdm(graphs):
                lg = g.line_graph(shared=True)
                lg.apply_edges(compute_bond_cosines)
                self.line_graphs.append(lg)

        if classification:
            self.labels = self.labels.view(-1).long()
            print("Classification dataset.", self.labels)

    @staticmethod
    def _get_attribute_lookup(atom_features: str = "cgcnn"):
        """Build a lookup array indexed by atomic number."""
        max_z = max(v["Z"] for v in chem_data.values())

        # get feature shape (referencing Carbon)
        template = get_node_attributes("C", atom_features)

        features = np.zeros((1 + max_z, len(template)))

        for element, v in chem_data.items():
            z = v["Z"]
            x = get_node_attributes(element, atom_features)

            if x is not None:
                features[z, :] = x

        return features

    def __len__(self):
        """Get length."""
        return self.labels.shape[0]

    def __getitem__(self, idx):
        """Get StructureDataset sample."""
        g = self.graphs[idx]
        label = self.labels[idx]
        lattice = self.lattices[idx]
        # id = self.ids[idx]
        if self.transform:
            g = self.transform(g)

        if self.line_graph:
            return g, self.line_graphs[idx], lattice, label

        return g, lattice, label

    def setup_standardizer(self, ids):
        """Atom-wise feature standardization transform."""
        x = torch.cat(
            [
                g.ndata["atom_features"]
                for idx, g in enumerate(self.graphs)
                if idx in ids
            ]
        )
        self.atom_feature_mean = x.mean(0)
        self.atom_feature_std = x.std(0)

        self.transform = Standardize(
            self.atom_feature_mean, self.atom_feature_std
        )

    @staticmethod
    def collate(samples: List[Tuple[dgl.DGLGraph, torch.Tensor]]):
        """Dataloader helper to batch graphs cross `samples`."""
        graphs, lattices, labels = map(list, zip(*samples))
        batched_graph = dgl.batch(graphs)
        return batched_graph, torch.tensor(lattices), torch.tensor(labels)

    @staticmethod
    def collate_line_graph(
        samples: List[Tuple[dgl.DGLGraph, dgl.DGLGraph, torch.Tensor]],
    ):
        """Dataloader helper to batch graphs cross `samples`."""
        graphs, line_graphs, lattices, labels = map(list, zip(*samples))
        batched_graph = dgl.batch(graphs)
        batched_line_graph = dgl.batch(line_graphs)
        if len(labels[0].size()) > 0:
            return batched_graph, batched_line_graph, torch.stack(labels)
        else:
            return (
                batched_graph,
                batched_line_graph,
                torch.stack(lattices),
                torch.tensor(labels),
            )


"""
if __name__ == "__main__":
    from jarvis.core.atoms import Atoms
    from jarvis.db.figshare import get_jid_data

    atoms = Atoms.from_dict(get_jid_data("JVASP-664")["atoms"])
    g = Graph.from_atoms(
        atoms=atoms,
        features="basic",
        get_prim=True,
        zero_diag=True,
        node_atomwise_angle_dist=True,
        node_atomwise_rdf=True,
    )
    g = Graph.from_atoms(
        atoms=atoms,
        features="cfid",
        get_prim=True,
        zero_diag=True,
        node_atomwise_angle_dist=True,
        node_atomwise_rdf=True,
    )
    g = Graph.from_atoms(
        atoms=atoms,
        features="atomic_number",
        get_prim=True,
        zero_diag=True,
        node_atomwise_angle_dist=True,
        node_atomwise_rdf=True,
    )
    g = Graph.from_atoms(atoms=atoms, features="basic")
    g = Graph.from_atoms(
        atoms=atoms, features=["Z", "atom_mass", "max_oxid_s"]
    )
    g = Graph.from_atoms(atoms=atoms, features="cfid")
    # print(g)
    d = g.to_dict()
    g = Graph.from_dict(d)
    num_nodes = g.num_nodes
    num_edges = g.num_edges
    print(num_nodes, num_edges)
    assert num_nodes == 48
    assert num_edges == 2304
    assert len(g.adjacency_matrix) == 2304
    # graph, color_map = get_networkx_graph(atoms)
    # nx.draw(graph, node_color=color_map, with_labels=True)
    # from jarvis.analysis.structure.neighbors import NeighborsAnalysis
"""
