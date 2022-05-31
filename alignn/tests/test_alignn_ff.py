from jarvis.db.figshare import get_jid_data
from jarvis.core.atoms import Atoms as JAtoms
from alignn.alignn_ff import ALIGNNFF_Calculator, RunMD, OptimizeAtoms

"""
def test_alignnff():
    atoms = JAtoms.from_dict(get_jid_data()["atoms"])
    print(atoms)
    model_path = (
        "/wrk/knc6/AlIGNN-FF/jdft_max_min_307113_epa/out/best_model.pt"
    )
    calculator = ALIGNNFF_Calculator(model_path=model_path)
    opt = OptimizeAtoms(
        optimizer="BFGS", calculator=calculator, jarvis_atoms=atoms
    ).optimize()
    print(opt)
    print("NVT")
    md = RunMD(calculator=calculator, jarvis_atoms=opt, nsteps=5).run()
    print("NPT")
    md = RunMD(
        calculator=calculator, ensemble="npt", jarvis_atoms=opt, nsteps=5
    ).run()


# test_alignnff()
"""
