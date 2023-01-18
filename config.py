from yacs.config import CfgNode as CN

_C = CN(new_allowed=True)

_C.DATASET = CN(new_allowed=True)
_C.DATASET.TRAIN_PATH = 'dataset/train/'
_C.DATASET.VALID_PATH = 'dataset/validation/'
_C.DATASET.TEST_PATH = 'dataset/test/'

_C.TRAIN = CN(new_allowed=True)
_C.TRAIN.IMG_DIM = 128
_C.TRAIN.NUM_CHANNELS = 3
_C.TRAIN.X_DIM = 128 * 128 * 3
_C.TRAIN.BATCH_SIZE = 32
_C.TRAIN.NUM_EPOCHS = 50
_C.TRAIN.LR = 0.001

_C.QUANTIZER = CN(new_allowed=True)
_C.QUANTIZER.EMBEDDING_DIM = 64
_C.QUANTIZER.NUM_EMBEDDINGS = 512
_C.QUANTIZER.LOG_VAR_Q = 0.0
_C.QUANTIZER.LOG_VAR_Q_SCALAR = 3.0

_C.QUANTIZER.TEMPERATURE = CN(new_allowed=True)
_C.QUANTIZER.TEMPERATURE.INIT = 1.0
_C.QUANTIZER.TEMPERATURE.DECAY = 0.00001
_C.QUANTIZER.TEMPERATURE.MIN = 0.0

_C.DEVICE = 'mps'
_C.CHECKPOINT_PATH = 'checkpoints/'
_C.SAMPLE_PATH = 'samples/'
_C.PRINT_PER_BATCH = 50


def get_cfg_defaults():
    return _C.clone()
