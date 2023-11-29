# Author: Henry Heitmann

class MyMessage(object):
    # server to dealer
    MSG_TYPE_S2D_RECOVER_MASKS = 1

    # client to dealer
    MSG_TYPE_C2D_SEND_MASK_TO_DEALER = 2

    # dealer to server
    MSG_TYPE_D2S_SEND_AGGREGATED_MASK_TO_SERVER = 3

    # server to client
    MSG_TYPE_S2C_INIT_CONFIG = 4
    MSG_TYPE_S2C_SYNC_MODEL_TO_CLIENT = 5

    # client to server
    MSG_TYPE_C2S_SEND_MODEL_TO_SERVER = 6

    MSG_ARG_KEY_TYPE = "msg_type"
    MSG_ARG_KEY_SENDER = "sender"
    MSG_ARG_KEY_RECEIVER = "receiver"

    """
        message payload keywords definition
    """
    MSG_ARG_KEY_NUM_SAMPLES = "num_samples"
    MSG_ARG_KEY_MODEL_PARAMS = "model_params"
    MSG_ARG_KEY_CLIENT_INDEX = "client_idx"

    MSG_ARG_KEY_TRAIN_CORRECT = "train_correct"
    MSG_ARG_KEY_TRAIN_ERROR = "train_error"
    MSG_ARG_KEY_TRAIN_NUM = "train_num_sample"

    MSG_ARG_KEY_TEST_CORRECT = "test_correct"
    MSG_ARG_KEY_TEST_ERROR = "test_error"
    MSG_ARG_KEY_TEST_NUM = "test_num_sample"

    MSG_ARG_KEY_CLIENT_ID_LIST = "list of active clients"
    MSG_ARG_KEY_DEALER_ID_LIST = "list of all used dealers"
    MSG_ARG_KEY_AGGREGATED_MASKS = "aggregated mask of all active clients"
    MSG_ARG_KEY_MASK = "mask of a single client"
    MSG_ARG_KEY_MODEL_SHAPE = "shape of the model"
