"""Module to visualize networkx graph."""
import networkx as nx
from jarvis.core.atoms import Atoms
from jarvis.core.graphs import Graph
from jarvis.db.figshare import get_jid_data

# import matplotlib.pyplot as plt

# %matplotlib inline
a = Atoms.from_dict(get_jid_data("JVASP-664")["atoms"])
g, lg = Graph.atom_dgl_multigraph(a)
nx.draw(lg.to_networkx())
nx.draw(g.to_networkx())
