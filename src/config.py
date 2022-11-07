import hydra
from omegaconf import OmegaConf
from pydantic import BaseModel

from utils_train import OptimizerType, SchedulerType


class DataModuleConfig(BaseModel):
    dataset_dirs: list[str]
    image_size: int
    # csv_spacing: float
    dataset_frac: float
    shuffle_before_splitting: bool
    num_workers: int
    drop_last: bool
    batch_size: int
    dataset_csv: str
    use_single_images: bool


class LogConfig(BaseModel):
    level: str
    name: str


class GeoAreaConfig(BaseModel):
    local_crs: int
    country_iso2: str


class GeoConfig(BaseModel):
    global_crs: int
    area: GeoAreaConfig


class PyanticHydraConfig(BaseModel):
    geo: GeoConfig
    datamodule: DataModuleConfig
    log: LogConfig
    datetime_format: str


hydra.initialize(version_base=None, config_path="../conf", job_name="app")
_hydra_dict_config = hydra.compose(config_name="config")
OmegaConf.resolve(_hydra_dict_config)
OmegaConf.set_readonly(_hydra_dict_config, True)
OmegaConf.set_struct(_hydra_dict_config, True)
_config_dict = OmegaConf.to_container(_hydra_dict_config)
cfg = PyanticHydraConfig(**_config_dict)  # type ignore


# """CSV"""
# DEFAULT_SPACING = 0.7

# """IMAGE"""
# DEFAULT_IMAGE_SIZE = 224
# DEFAULT_IMAGE_MEAN, DEFAULT_IMAGE_STD = [0.5006, 0.5116, 0.4869], [
#     0.1966,
#     0.1951,
#     0.2355,
# ]

# """DATASET"""
# DEFAULT_DATASET_FRAC = 1
# DEFAULT_TRAIN_FRAC = 0.9
# DEFAULT_VAL_FRAC = 0.05
# DEFAULT_TEST_FRAC = 0.05
# DEAFULT_shuffle_before_splitting = False

# """LOGGING"""
# LOG_EVERY_N = 100

# """MODEL"""
# DEFAULT_PRETRAINED = True
# DEFAULT_MODEL = "resnext101_32x8d"
# DEFAULT_UNFREEZE_LAYERS_NUM = "all"

# """TRAINING"""
# DEFAULT_EPOCHS = 22

# """OPTIM"""

# DEFAULT_LR = 2e-5
# DEFAULT_LR_FINETUNE = 2e-5
# DEAFULT_NUM_WORKERS = 4
# DEAFULT_DROP_LAST = True
# DEFAULT_BATCH_SIZE = 8
# DEFAULT_VAL_CHECK_EVERY_N_EPOCH = 1
# DEFAULT_FINETUNING_EPOCH_PERIOD = 5
# DEFAULT_EARLY_STOPPING_EPOCH_FREQ = 15
# DEFAULT_WEIGHT_DECAY = 0
# DEFAULT_SCHEDULER = SchedulerType.PLATEAU.value
# DEFAULT_OPTIMIZER = OptimizerType.ADAM.value

# DEFAULT_CROATIA_CRS = 3766
# DEFAULT_GLOBAL_CRS = 4326
# DEFAULT_COUNTRY_ISO2 = "HR"

# DEFAULT_TORCHVISION_VERSION = "pytorch/vision:v0.12.0"

# if __name__ == "__main__":
#     pass
