"""Module to compare JARVIS-CFID results."""
# from jarvis.ai.pkgs.utils import get_ml_data
# from jarvis.ai.pkgs.utils import regr_scores
from jarvis.db.figshare import data as jdata
import random
import numpy as np
import math
import lightgbm as lgb
from jarvis.ai.pkgs.utils import regr_scores

props = [
    "formation_energy_peratom",
    "optb88vdw_bandgap",
    "bulk_modulus_kv",
    "shear_modulus_gv",
    "mbj_bandgap",
    "slme",
    "magmom_oszicar",
    "spillage",
    "kpoint_length_unit",
    "encut",
    "optb88vdw_total_energy",
    "epsx",
    "epsy",
    "epsz",
    "mepsx",
    "mepsy",
    "mepsz",
    "max_ir_mode",
    "avg_elec_mass",
    "avg_hole_mass",
    "max_efg",
    "min_ir_mode",
    "n-Seebeck",
    "p-Seebeck",
    "n-powerfact",
    "p-powerfact",
    "ncond",
    "pcond",
    "nkappa",
    "pkappa",
    "ehull",
    "exfoliation_energy",
    "dfpt_piezo_max_dielectric",
    "dfpt_piezo_max_eij",
    "dfpt_piezo_max_dij",
]


def load_dataset(
    name: str = "dft_3d",
    target=None,
    n_train=None,
    n_val=None,
    n_test=None,
    train_ratio=None,
    val_ratio=0.1,
    test_ratio=0.1,
    split_seed=123,
):
    """Load jarvis data."""
    d = jdata(name)
    y = []
    X = []
    for i in d:
        if (
            i[target] != "na"
            and not math.isnan(i[target])
            and len(i["desc"]) == 1557
        ):
            X.append(i["desc"])
            y.append(i[target])
    X = np.array(X)
    y = np.array(y)
    total_size = len(X)
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
    ids = np.arange(total_size)

    random.seed(split_seed)
    random.shuffle(ids)
    if n_train + n_val + n_test > total_size:
        raise ValueError("Check total number of samples.")
    print("n_train,n_test,total_size", n_train, n_test, total_size)
    id_train = ids[:n_train]
    id_val = ids[-(n_val + n_test) : -n_test]  # noqa:E203
    id_test = ids[-n_test:]
    print("id_train,id_val,id_test", len(id_train), len(id_val), len(id_test))
    return X[id_train], y[id_train], X[id_test], y[id_test]


lgbm = lgb.LGBMRegressor(
    # device="gpu",
    n_estimators=1740,
    learning_rate=0.040552334327414057,
    num_leaves=291,
    max_depth=16,
    min_data_in_leaf=14,
)

lgbm = lgb.LGBMRegressor(
    # device="gpu",
    n_estimators=1170,
    learning_rate=0.15375236057119931,
    num_leaves=273,
)
for i in props:
    X_train, y_train, X_test, y_test = load_dataset(name="dft_3d", target=i)
    lgbm = lgb.LGBMRegressor(
        # device="gpu",
        n_estimators=1170,
        learning_rate=0.15375236057119931,
        num_leaves=273,
    )
    lgbm.fit(X_train, y_train)
    pred = lgbm.predict(X_test)
    reg_sc = regr_scores(y_test, pred)
    print(i, reg_sc["mae"])
