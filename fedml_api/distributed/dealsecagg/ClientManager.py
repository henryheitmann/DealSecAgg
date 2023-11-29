# Author: Henry Heitmann

import logging
import random

from Crypto.Random import get_random_bytes

from fedml_api.distributed.utils.Benchmark import Benchmark
from fedml_api.distributed.utils.masking import generate_private_mask, add_models
from fedml_api.distributed.utils.utils import convert_int8_to_float32, convert_int16_to_float32, \
    transform_tensor_to_array, convert_float32_to_int16, convert_float32_to_int8, transform_array_to_tensor
from fedml_core.distributed.client.client_manager import ClientManager
from fedml_core.distributed.communication.message import Message
from .message_define import MyMessage


class ClientManager(ClientManager):
    def __init__(self, args, trainer, comm=None, rank=0, size=0, backend="MPI"):
        super().__init__(args, comm, rank , size, backend)
        self.benchmark = Benchmark()
        self.trainer = trainer
        self.train_model_rounds = args.train_model_rounds
        self.num_rounds = args.comm_round
        self.round_idx = 0
        self.worker_num = size - 1
        self.num_dropouts = args.num_dropouts
        self.quantization_datatype = args.quantization
        self.total_num_dealers = args.total_num_dealers
        self.num_dealers = args.num_dealers
        self.dealer_ids = [i for i in range(1, self.total_num_dealers+1)]

    def run(self):
        super().run()

    def register_message_receive_handlers(self):
        self.register_message_receive_handler(
            MyMessage.MSG_TYPE_S2C_INIT_CONFIG, self.handle_message_init
        )
        self.register_message_receive_handler(
            MyMessage.MSG_TYPE_S2C_SYNC_MODEL_TO_CLIENT, self.handle_message_sync_model
        )

    def handle_message_init(self, msg_params):
        self.benchmark.set_start_time()
        logging.info("client %d handle_message_init from server." % self.get_sender_id())
        global_model_params = msg_params.get(MyMessage.MSG_ARG_KEY_MODEL_PARAMS)
        client_index = msg_params.get(MyMessage.MSG_ARG_KEY_CLIENT_INDEX)

        if self.quantization_datatype == 'int8':
            global_model_params = convert_int8_to_float32(global_model_params)
        elif self.quantization_datatype == 'int16':
            global_model_params = convert_int16_to_float32(global_model_params)
        model_tensor = transform_array_to_tensor(global_model_params)

        self.trainer.update_model(model_tensor)
        self.trainer.update_dataset(int(client_index))
        self.round_idx = 0
        self.benchmark.add_offline_time()
        self.__train()

    def handle_message_sync_model(self, msg_params):
        self.benchmark.set_start_time()
        logging.info("client %d handle_message_sync_model from server." % self.get_sender_id())
        global_model_params = msg_params.get(MyMessage.MSG_ARG_KEY_MODEL_PARAMS)
        client_index = msg_params.get(MyMessage.MSG_ARG_KEY_CLIENT_INDEX)

        if self.quantization_datatype == 'int8':
            global_model_params = convert_int8_to_float32(global_model_params)
        elif self.quantization_datatype == 'int16':
            global_model_params = convert_int16_to_float32(global_model_params)
        model_tensor = transform_array_to_tensor(global_model_params)

        self.trainer.update_model(model_tensor)
        self.trainer.update_dataset(int(client_index))
        self.round_idx += 1
        self.benchmark.add_offline_time()
        self.__train()

    def __train(self):
        self.benchmark.set_last_time()
        is_dropout = self.check_is_drop_out()
        if is_dropout:
            logging.info("Client %d has dropped out" % self.get_sender_id())
        else:
            logging.info("Client %d #######training########### round_id = %d" % (self.get_sender_id(), self.round_idx))
            local_sample_num = 0
            weights = self.trainer.get_model_params()

            for i in range(0, self.train_model_rounds):
                self.trainer.update_model(weights)
                weights, local_sample_num = self.trainer.train(self.round_idx)

            self.benchmark.add_training_time()

            weights = transform_tensor_to_array(weights)
            if self.quantization_datatype == 'int8':
                logging.info("quantization to int8")
                weights = convert_float32_to_int8(weights)
            elif self.quantization_datatype == 'int16':
                logging.info("quantization to int16")
                weights = convert_float32_to_int16(weights)

            # Mask the local model
            masked_weights = weights

            random.seed(self.get_sender_id() + self.round_idx)
            selected_dealers = random.sample(self.dealer_ids, self.num_dealers)

            for dealer in selected_dealers:
                private_seed = get_random_bytes(16)
                mask = generate_private_mask(private_seed, weights, self.quantization_datatype)
                masked_weights = add_models(masked_weights, mask)
                self.send_mask_to_dealer(dealer, private_seed)

            self.send_model_to_server(0, masked_weights, local_sample_num, selected_dealers)
            self.benchmark.add_masking_time()
            self.benchmark.set_total_bits()
            self.benchmark.set_total_time()
            self.benchmark.write_benchmark(
                path=self.args.model + '_' + str(self.args.client_num_in_total) + '_'
                     + str(self.args.client_num_per_round) + '_' + str(self.args.num_dropouts) + '_'
                     + self.args.quantization + '_' + str(self.args.total_num_dealers) + '_'
                     + str(self.args.num_dealers),
                filename='client')


    def send_mask_to_dealer(self, receiver_id, private_seed):
        message = Message(MyMessage.MSG_TYPE_C2D_SEND_MASK_TO_DEALER, self.get_sender_id(), receiver_id)
        message.add_params(MyMessage.MSG_ARG_KEY_MASK, private_seed)
        message_size = message.get_message_size()
        self.benchmark.add_masking_bits(message_size)
        self.send_message(message)

    def send_model_to_server(self, receiver_id, weights, local_sample_num, dealer_ids):
        message = Message(MyMessage.MSG_TYPE_C2S_SEND_MODEL_TO_SERVER, self.get_sender_id(), receiver_id)
        message.add_params(MyMessage.MSG_ARG_KEY_MODEL_PARAMS, weights)
        message.add_params(MyMessage.MSG_ARG_KEY_NUM_SAMPLES, local_sample_num)
        message.add_params(MyMessage.MSG_ARG_KEY_DEALER_ID_LIST, dealer_ids)
        message_size = message.get_message_size()
        self.benchmark.add_masking_bits(message_size)
        self.send_message(message)

    def check_is_drop_out(self):
        if self.get_sender_id() > self.worker_num - self.num_dropouts:
            return True
        return False
