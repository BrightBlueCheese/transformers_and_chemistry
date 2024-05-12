from deepchem.molnet import *

# refered from https://github.com/seyonechithrananda/bert-loves-chemistry
molnet_dict = {
    "bace_classification": {
        "dataset_type": "classification",
        "load_fn": load_bace_classification,
        "split": "scaffold",
    },
    "bace_regression": {
        "dataset_type": "regression",
        "load_fn": load_bace_regression,
        "split": "scaffold",
    },
    "delaney": {
        "dataset_type": "regression",
        "load_fn": load_delaney,
        "split": "scaffold",
    },
    "hiv": {
        "dataset_type": "classification",
        "load_fn": load_hiv,
        "split": "scaffold",
    },
    "lipo": {
        "dataset_type": "regression",
        "load_fn": load_lipo,
        "split": "scaffold",
    },
    # "qm7": {
    #     "dataset_type": "regression",
    #     "load_fn": load_qm7,
    #     "split": "stratified",
    # },
    # "qm8": {
    #     "dataset_type": "regression",
    #     "load_fn": load_qm8,
    #     "split": "random",
    # },
    # "qm9_mu": {
    #     "dataset_type": "regression",
    #     "load_fn": load_qm9,
    #     "split": "random",
    #     "specific_task": ["mu"],
    # },
    # "qm9_g298": {
    #     "dataset_type": "regression",
    #     "load_fn": load_qm9,
    #     "split": "random",
    #     "specific_task": ["g298"],
    # },
    # "sider": {
    #     "dataset_type": "classification",
    #     "load_fn": load_sider,
    #     "split": "scaffold",
    # },
    # "tox21_nr_ar": {
    #     "dataset_type": "classification",
    #     "load_fn": load_tox21,
    #     "split": "random",
    #     "specific_task": ["NR-AR"],
    # },
    "tox21_sr_p53": {
        "dataset_type": "classification",
        "load_fn": load_tox21,
        "split": "random",
        "specific_task": ["SR-p53"],
    },
}