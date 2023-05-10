import json
import torch
from torch import nn
from torch.nn import functional as F
from jarvis.core.atoms import Atoms

import dgl
import dgl.function as fn
from dgl.nn import SumPooling

from alignn.models.alignn import EdgeGatedGraphConv

# double precision for gradient checking
torch.set_default_dtype(torch.float64)

jvasp_98225_data = {
    "lattice_mat": [
        [7.2963518353359165, 0.0, 0.0],
        [0.0, 12.357041682775112, -5.2845586858227165],
        [0.0, -0.0387593429993432, 14.224638121809875],
    ],
    "coords": [
        [0.915242849989437, 6.855852122669934, 9.072691488805011],
        [4.624926092686879, 4.410242674901673, 10.692815115124906],
        [6.319598907313122, 10.588762674901673, 8.050535115124907],
        [2.6714239073131205, 7.908037325098326, -1.7527351151248998],
        [2.160203276580653, 0.46740047015408986, 5.445837057504562],
        [5.808378276580651, 5.672359529845909, 6.136522942495441],
        [5.136146723419349, 11.850879529845914, 3.4942429424954384],
        [1.4879717234193466, 6.645920470154087, 2.8035570575045634],
        [0.9767510926868789, 1.7295173250983271, 0.8895448848751],
        [4.149189130241182, 3.7835217605140947, 3.289715069117776],
        [6.795335869758818, 9.962041760514095, 0.6474350691177748],
        [3.147160869758816, 8.534758239485901, 5.650364930882219],
        [2.732932150010564, 0.6773321226699319, 11.714971488805013],
        [6.3811071500105605, 5.462427877330069, -0.13261148880500784],
        [4.56341784998944, 11.640947877330065, -2.774891488805006],
        [0.5010141302411835, 2.3562382394859105, 8.292644930882215],
        [2.57977406359738, 10.129401145455988, 8.824521146549861],
        [6.227949063597383, 8.367398854544012, -2.5267211465498614],
        [2.8191403695947996, 4.588088605622105, -0.30548283747092936],
        [6.467315369594803, 1.5516713943778946, 11.887842837470933],
        [4.477209630405198, 7.730191394377894, 9.245562837470933],
        [0.8290346304051991, 10.766608605622105, -2.947762837470929],
        [0.44432405371819356, 3.3442489732904943, 4.068158750100403],
        [3.2038509462818063, 9.5227689732905, 1.4258787501003996],
        [6.852025946281809, 8.974031026709499, 4.871921249899597],
        [2.125497471695336, 4.814663054405411, 6.181902909761823],
        [5.773672471695338, 1.3250969455945911, 5.40045709023818],
        [5.170852528304662, 7.503616945594589, 2.7581770902381804],
        [1.5226775283046639, 10.99318305440541, 3.5396229097618233],
        [1.0684009364026184, 3.950881145455989, 11.46680114654986],
        [4.092499053718192, 2.7955110267095065, 7.514201249899595],
        [4.716575936402617, 2.188878854544011, 0.11555885345013912],
    ],
    "elements": [
        "K",
        "K",
        "K",
        "K",
        "K",
        "K",
        "K",
        "K",
        "K",
        "K",
        "K",
        "K",
        "K",
        "K",
        "K",
        "K",
        "Bi",
        "Bi",
        "Bi",
        "Bi",
        "Bi",
        "Bi",
        "Bi",
        "Bi",
        "Bi",
        "Bi",
        "Bi",
        "Bi",
        "Bi",
        "Bi",
        "Bi",
        "Bi",
    ],
    "abc": [7.29635, 13.439606, 14.224693],
    "angles": [113.3104, 90.0, 90.0],
    "cartesian": True,
    "props": [
        "",
        "",
        "",
        "",
        "",
        "",
        "",
        "",
        "",
        "",
        "",
        "",
        "",
        "",
        "",
        "",
        "",
        "",
        "",
        "",
        "",
        "",
        "",
        "",
        "",
        "",
        "",
        "",
        "",
        "",
        "",
        "",
    ],
}
at = Atoms.from_dict(jvasp_98225_data)


class SimpleModel(nn.Module):
    """Simplified GNN depending only on positions"""

    def __init__(self, cutoff=8, width=16):
        super().__init__()
        self.cutoff = cutoff
        self.width = width

        self.edge_embedding = nn.Linear(1, width)
        self.hidden1 = EdgeGatedGraphConv(width, width)
        self.hidden2 = EdgeGatedGraphConv(width, width)
        self.fc = nn.Linear(width, 1)

        self.readout = SumPooling()

    def forward(self, positions, autograd_forces=False):

        # make sure positions are included in the autograd graph
        if autograd_forces:
            positions.requires_grad_(True)

        # non-periodic radius graph construction
        g = dgl.radius_graph(positions, self.cutoff)
        g.ndata["r"] = positions

        # compute bond displacement vectors
        g.apply_edges(fn.v_sub_u("r", "r", "bondvec"))
        bondvec = g.edata.pop("bondvec")
        bondlength = torch.norm(bondvec, dim=1).squeeze()

        # expand bond length basis functions
        y = self.edge_embedding(bondlength.unsqueeze(-1))
        g.edata["y"] = y

        # constant node features
        x = torch.ones(g.num_nodes(), self.width)

        # graph convolution layers
        x, y = self.hidden1(g, x, y)
        x, y = self.hidden2(g, x, y)

        # node-wise prediction
        energy = self.fc(x)

        # reduction
        total_energy = torch.squeeze(self.readout(g, energy))

        if not autograd_forces:
            return total_energy

        # force calculation based on position
        # retain graph for displacement-based grad calculation
        forces_x = -torch.autograd.grad(
            total_energy, positions, retain_graph=True
        )[0]

        # force calculation based on displacements
        # gives dE / d{vec{r}_{ij}}
        # want dE / d{r_i}
        # combine r_{ji} and r_{ij}
        pairwise_forces = -torch.autograd.grad(total_energy, bondvec)[0]

        # reduce over bonds to get forces on each atom
        g.edata["pairwise_forces"] = pairwise_forces
        g.update_all(
            fn.copy_e("pairwise_forces", "m"), fn.sum("m", "forces_ji")
        )

        # reduce over reverse edges too!
        rg = dgl.reverse(g, copy_edata=True)
        rg.update_all(
            fn.copy_e("pairwise_forces", "m"), fn.sum("m", "forces_ij")
        )

        forces_vec = torch.squeeze(
            g.ndata["forces_ji"] - rg.ndata["forces_ij"]
        )

        return total_energy, forces_x, forces_vec


def test_compare_position_and_displacement_autograd_forces():
    """Check that all elements of both autograd force methods match."""
    torch.set_default_dtype(torch.float64)
    model = SimpleModel(cutoff=5)

    # evaluate energy and both styles of autograd forces
    x = torch.from_numpy(at.cart_coords)
    e, f_x, f_vec = model(x, autograd_forces=True)

    # check that all elements of both autograd force methods
    # are equal to within numerical precision
    assert torch.isclose(f_x, f_vec).all().item()


def test_compare_forces_finite_difference():
    """Compare autograd forces with centered finite difference."""
    torch.set_default_dtype(torch.float64)

    model = SimpleModel(cutoff=5)
    x = torch.from_numpy(at.cart_coords)

    def finite_difference_force(x, i, j, delta=1e-6):
        """compute force for atom i, component j with centered finite difference"""
        xa = x.detach().clone()
        xb = x.detach().clone()
        xa[i, j] -= delta
        xb[i, j] += delta

        with torch.no_grad():
            force_diff = -(model(xb) - model(xa)) / (2 * delta)

        return force_diff

    # evaluate forces with autograd
    e, f_x, f_vec = model(x, autograd_forces=True)

    # construct array of numerical forces with a double list comprehension
    f_dx = torch.tensor(
        [
            [finite_difference_force(x, i, j) for j in range(3)]
            for i in range(at.num_atoms)
        ]
    )

    # compare to autograd using the numerical parameters from torch.autograd.gradcheck
  
    assert torch.isclose(f_vec, f_dx, atol=1e-05, rtol=0.001).all().item()
    assert torch.isclose(f_x, f_dx, atol=1e-05, rtol=0.001).all().item()
