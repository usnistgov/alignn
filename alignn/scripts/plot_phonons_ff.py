"""Module for phonons using ase."""
# from ase.build import bulk
# from ase.calculators.emt import EMT
from ase.phonons import Phonons
import matplotlib.pyplot as plt  # noqa

# from alignn.ff.ff import AlignnAtomwiseCalculator, default_path, ev_curve
from alignn.ff.ff import AlignnAtomwiseCalculator, phonons, ase_phonon
from jarvis.analysis.structure.spacegroup import (
    Spacegroup3D,
    #    symmetrically_distinct_miller_indices,
)
from jarvis.core.atoms import Atoms as JarvisAtoms
from jarvis.db.figshare import get_jid_data
from ase.cell import Cell


if __name__ == "__main__":
    model_path = "/wrk/knc6/ALINN_FC/FD_mult/temp_new"
    phonons(atoms=atoms, model_path=mlearn, enforce_c_size=3)
    """
    bs = ase_phonon(
        jid="JVASP-32", ev_file="JVASP-32_ev.png", model_path=model_path
    )
    bs = ase_phonon(
        jid="JVASP-1002", ev_file="JVASP-32_ev.png", model_path=model_path
    )
    bs = ase_phonon(
        jid="JVASP-19821", ev_file="JVASP-19821_ev.png", model_path=model_path
    )
    bs = ase_phonon(
        jid="JVASP-254", ev_file="JVASP-254_ev.png", model_path=model_path
    )
    bs = ase_phonon(
        jid="JVASP-816", ev_file="JVASP-816_ev.png", model_path=model_path
    )
    for i in jids:
        try:
            ev_file = i + "_ev.png"
            bs = ase_phonon(jid=i, ev_file=ev_file, model_path=model_path)
        except:
            pass
    """
    # bs = ase_phonon(jid="JVASP-21195", ev_file="ev.png")
    # bs = ase_phonon(jid="JVASP-943", ev_file="ev.png")
    # bs = ase_phonon(jid="JVASP-1002", ev_file="ev.png")
    # bs = ase_phonon(jid="JVASP-816", ev_file="ev.png")
# bs = ase_phonon(jid="JVASP-1002")
