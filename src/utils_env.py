"""IMAGE"""
DEFAULT_IMAGE_SIZE = 224

"""DATASET"""
DEFAULT_DATASET_SIZE = 1
DEFAULT_TRAIN_FRAC = 0.6
DEFAULT_VAL_FRAC = 0.2
DEFAULT_TEST_FRAC = 0.2
DEAFULT_SHUFFLE_DATASET_BEFORE_SPLITTING = False

"""LOGGING"""
LOG_EVERY_N = 10
LOG_PRINT_EVERY_N = 5

"""MODEL"""
DEFAULT_PRETRAINED = True
DEFAULT_MODEL = "resnext101_32x8d"
DEFAULT_UNFREEZE_LAYERS_NUM = "all"

"""TRAINING"""
DEFAULT_LR = 0.0001
DEFAULT_EPOCHS = 40
DEAFULT_NUM_WORKERS = 4
DEAFULT_DROP_LAST = True
DEFAULT_BATCH_SIZE = 8
DEFAULT_VAL_CHECK_EVERY_N_EPOCH = 1
DEFAULT_EARLY_STOPPING_EPOCH_FREQ = 8
DEFAULT_FINETUNING_EPOCH_PERIOD = 4
DEFAULT_WEIGHT_DECAY = 0.00001

if __name__ == "__main__":
    pass