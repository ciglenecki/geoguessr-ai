"""CSV"""
from utils_train import SchedulerType


DEFAULT_SPACING = 0.5

"""IMAGE"""
DEFAULT_IMAGE_SIZE = 112

"""DATASET"""
DEFAULT_DATASET_SIZE = 1
DEFAULT_TRAIN_FRAC = 0.8
DEFAULT_VAL_FRAC = 0.1
DEFAULT_TEST_FRAC = 0.1
DEAFULT_SHUFFLE_DATASET_BEFORE_SPLITTING = False
DEFAULT_LOAD_DATASET_IN_RAM = False

"""LOGGING"""
LOG_EVERY_N = 20
LOG_PRINT_EVERY_N = 5

"""MODEL"""
DEFAULT_PRETRAINED = True
DEFAULT_MODEL = "resnext101_32x8d"
DEFAULT_UNFREEZE_LAYERS_NUM = "all"

"""TRAINING"""
DEFAULT_LR = 0.1
DEFAULT_EPOCHS = 35
DEAFULT_NUM_WORKERS = 4
DEAFULT_DROP_LAST = True
DEFAULT_BATCH_SIZE = 8
DEFAULT_VAL_CHECK_EVERY_N_EPOCH = 1
DEFAULT_FINETUNING_EPOCH_PERIOD = 5
DEFAULT_EARLY_STOPPING_EPOCH_FREQ = 12
DEFAULT_WEIGHT_DECAY = 0
DEFAULT_SCHEDULER = SchedulerType.PLATEAU.value

DEFAULT_CROATIA_CRS = 3766
DEFAULT_GLOBAL_CRS = 4326

DEFAULT_TORCHVISION_VERSION = "pytorch/vision:v0.12.0"
if __name__ == "__main__":
    pass
