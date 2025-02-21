# specify the root location where u downloaded the dataset
root_location = "/CMU_16825_Work/CMU_16825_Learning_For_3D_Vision/proj2"
use_full_dataset = False
use_03001627_set = False
dataset_name = (
    "r2n2_shapenet_dataset_full" if use_full_dataset else "r2n2_shapenet_dataset"
)

R2N2_PATH = f"{root_location}/{dataset_name}/r2n2"
SHAPENET_PATH = f"{root_location}/{dataset_name}/shapenet"

if use_full_dataset:
    if use_03001627_set:
        SPLITS_PATH = f"{root_location}/{dataset_name}/split_03001627.json"  # split file contains data entry for 03001627 class
    else:
        SPLITS_PATH = f"{root_location}/{dataset_name}/split_3c.json"  # split file contains data entry for 3 classes
else:
    SPLITS_PATH = f"{root_location}/{dataset_name}/split_03001627.json"  # split file contains data entry for 03001627 class
