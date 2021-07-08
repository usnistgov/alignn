"""Module to visualize networkx graph."""
from jarvis.db.figshare import get_jid_data
from jarvis.core.atoms import Atoms
from jarvis.core.graphs import Graph
import networkx as nx
# import matplotlib.pyplot as plt

# %matplotlib inline
a = Atoms.from_dict(get_jid_data("JVASP-664")["atoms"])
g, lg = Graph.atom_dgl_multigraph(a)
nx.draw(lg.to_networkx())
nx.draw(g.to_networkx())
