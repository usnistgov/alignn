#!pip install dgl==0.4.3 jarvis-tools==2021.2.3 pymatgen==2018.9.30
def dgl_crystal_pmg(structure,primitive = False,cutoff = 8,supercell_size= 1) :
    # TODO: check lattice parameters and tile only if some cutoff radius is exceeded
    g = dgl.DGLGraph()

    if primitive:
      structure = structure.get_primitive_structure()

    if supercell_size > 1:
      structure.make_supercell(supercell_size, to_unit_cell=True)

    dist = structure.distance_matrix
    dist[dist > cutoff] = 0
    D = nx.Graph(dist)

    g.from_networkx(D, edge_attrs=['weight'])
    g.edata['bondlength'] = g.edata['weight']
    del g.edata['weight']
    g.ndata['atomic_number'] = np.array(structure.atomic_numbers, dtype=np.int8)
    return g
    
def dgl_crystal_jarvis(atoms,primitive = False,cutoff = 8,supercell_size= 1) :
    # TODO: check lattice parameters and tile only if some cutoff radius is exceeded
    g = dgl.DGLGraph()

    if primitive:
      atoms = atoms.get_primitive_atoms

    if supercell_size > 1:
      atoms.make_supercell(supercell_size)

    dist = atoms.raw_distance_matrix
    dist[dist > cutoff] = 0
    D = nx.Graph(dist)

    g.from_networkx(D, edge_attrs=['weight'])
    g.edata['bondlength'] = g.edata['weight']
    del g.edata['weight']
    g.ndata['atomic_number'] = np.array(atoms.atomic_numbers, dtype=np.int8)
    return g
