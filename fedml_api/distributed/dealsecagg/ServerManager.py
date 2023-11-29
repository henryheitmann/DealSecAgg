# Author: Henry Heitmann

import logging
from time import sleep

from fedml_api.distributed.utils.Benchmark import Benchmark
from fedml_api.distributed.utils.utils import transform_tensor_to_array, convert_float32_to_int16, \
    convert_float32_to_int8, get_model_shape
from fedml_core.distributed.communication.message import Message
from fedml_core.distributed.server.server_manager import ServerManager
from .message_define import MyMessage


class ServerManager(ServerManager):
    def __init__(
            self,
            args,
            aggregator,
            comm=None,
            rank=0,
            size=0,
            backend="MPI",
            is_preprocessed=False,
            preprocessed_client_lists=None
    ):
        super().__init__(args, comm, rank, size, backend)
        self.benchmark = Benchmark()
        self.args = args
        self.quantization_datatype = args.quantization
        self.aggregator = aggregator
        self.round_num = args.comm_round
        self.train_model_rounds = args.train_model_rounds
        self.round_idx = 0
        self.is_preprocessed = is_preprocessed
        self.preprocessed_client_lists = preprocessed_client_lists
        self.dropout_num = args.num_dropouts
        self.clients_current_round = []
        self.active_clients = []
        self.client_dealer_map = dict()
        self.dealer_client_map = dict()
        self.aggregated_masks = []
        self.total_dealer_number = args.total_num_dealers

    def run(self):
        super().run()

    def send_init_msg(self):
        self.benchmark.set_start_time()
        client_indexes = self.aggregator.client_sampling(
            self.round_idx, self.args.client_num_in_total, self.args.client_num_per_round
        )
        self.clients_current_round = client_indexes
        global_model_params = self.aggregator.get_global_model_params()
        model_array = transform_tensor_to_array(global_model_params)

        if self.quantization_datatype == 'int8':
            model_array = convert_float32_to_int8(model_array)
        elif self.quantization_datatype == 'int16':
            model_array = convert_float32_to_int16(model_array)

        for process_id in range(self.total_dealer_number + 1, self.size):
            self.send_message_init_config(process_id, model_array, client_indexes[process_id - self.total_dealer_number - 1])
        self.benchmark.add_offline_time()

    def register_message_receive_handlers(self):
        self.register_message_receive_handler(
            MyMessage.MSG_TYPE_C2S_SEND_MODEL_TO_SERVER, self.handle_message_receive_model
        )
        self.register_message_receive_handler(
            MyMessage.MSG_TYPE_D2S_SEND_AGGREGATED_MASK_TO_SERVER, self.handle_message_receive_mask
        )

    def handle_message_receive_model(self, msg_params):
        sender_id = msg_params.get(MyMessage.MSG_ARG_KEY_SENDER)
        model_params = msg_params.get(MyMessage.MSG_ARG_KEY_MODEL_PARAMS)
        local_sample_number = msg_params.get(MyMessage.MSG_ARG_KEY_NUM_SAMPLES)
        dealer_ids = msg_params.get(MyMessage.MSG_ARG_KEY_DEALER_ID_LIST)

        self.aggregator.add_local_trained_result(sender_id - 1, model_params, local_sample_number)
        self.active_clients.append(sender_id)
        self.client_dealer_map[sender_id] = dealer_ids
        b_all_received = self.check_whether_all_model_received()
        logging.info("Server: model_all_received = " + str(b_all_received) + " in round_idx %d" % self.round_idx)
        if b_all_received:
            self.benchmark.set_last_time()
            logging.info("Number of Dropouts: {}".format(self.dropout_num))

            for key in self.client_dealer_map.keys():
                for dealer in self.client_dealer_map[key]:
                    if dealer in self.dealer_client_map:
                        self.dealer_client_map[dealer].append(key)
                    else:
                        self.dealer_client_map[dealer] = [key]

            model_shape = get_model_shape(model_params)

            for dealer in self.dealer_client_map:
                self.send_message_recover_masks(dealer, self.dealer_client_map[dealer], model_shape)

            self.benchmark.add_masking_time()

    def handle_message_receive_mask(self, msg_params):
        sender_id = msg_params.get(MyMessage.MSG_ARG_KEY_SENDER)
        mask = msg_params.get(MyMessage.MSG_ARG_KEY_AGGREGATED_MASKS)

        logging.info("Server received mask from dealer %d" % sender_id)

        self.aggregated_masks.append(mask)

        all_received = self.check_whether_all_masks_received()
        logging.info("Server: mask_all_received = " + str(all_received) + " in round_idx %d" % self.round_idx)

        if all_received:
            self.benchmark.set_last_time()
            self.aggregator.add_masks(self.aggregated_masks)
            global_model_params = self.aggregator.aggregate_model_reconstruction()

            self.benchmark.add_unmasking_time()
            self.benchmark.set_total_time()
            self.benchmark.set_total_bits()
            self.benchmark.write_benchmark(
                path=self.args.model + '_' + str(self.args.client_num_in_total) + '_'
                     + str(self.args.client_num_per_round) + '_' + str(self.args.num_dropouts) + '_'
                     + self.args.quantization + '_' + str(self.args.total_num_dealers) + '_'
                     + str(self.args.num_dealers),
                filename='server')

            # evaluation
            if self.train_model_rounds != 0:
                self.aggregator.test_on_server_for_all_clients(self.round_idx)

            # start the next round
            self.benchmark.set_start_time()
            self.round_idx += 1
            self.active_clients.clear()
            self.dealer_client_map.clear()
            self.client_dealer_map.clear()
            self.aggregated_masks.clear()

            if self.round_idx == self.round_num:
                logging.info("=================TRAINING IS FINISHED!=============")
                sleep(3)
                self.finish()
            if self.is_preprocessed:
                if self.preprocessed_client_lists is None:
                    # sampling has already been done in data preprocessor
                    client_indexes = [self.round_idx] * self.args.client_num_per_round
                else:
                    client_indexes = self.preprocessed_client_lists[self.round_idx]
            else:
                # sampling clients
                client_indexes = self.aggregator.client_sampling(
                    self.round_idx, self.args.client_num_in_total, self.args.client_num_per_round
                )

            self.clients_current_round = client_indexes

            print("indexes of clients: " + str(client_indexes))
            print("size = %d" % self.size)

            if self.quantization_datatype == 'int8':
                global_model_params = convert_float32_to_int8(global_model_params)
            elif self.quantization_datatype == 'int16':
                global_model_params = convert_float32_to_int16(global_model_params)

            for receiver_id in range(self.total_dealer_number + 1, self.size):
                self.send_message_sync_model_to_client(
                    receiver_id, global_model_params, client_indexes[receiver_id - self.total_dealer_number - 1]
                )
            self.benchmark.add_offline_time()

    def send_message_init_config(self, receive_id, global_model_params, client_index):
        message = Message(MyMessage.MSG_TYPE_S2C_INIT_CONFIG, self.get_sender_id(), receive_id)
        message.add_params(MyMessage.MSG_ARG_KEY_MODEL_PARAMS, global_model_params)
        message.add_params(MyMessage.MSG_ARG_KEY_CLIENT_INDEX, str(client_index))
        message_size = message.get_message_size()
        self.benchmark.add_offline_bits(message_size)
        self.send_message(message)

    def send_message_recover_masks(self, receiver_id, active_clients, model_shape):
        message = Message(MyMessage.MSG_TYPE_S2D_RECOVER_MASKS, self.get_sender_id(), receiver_id)
        message.add_params(MyMessage.MSG_ARG_KEY_CLIENT_ID_LIST, active_clients)
        message.add_params(MyMessage.MSG_ARG_KEY_MODEL_SHAPE, model_shape)
        message_size = message.get_message_size()
        self.benchmark.add_unmasking_bits(message_size)
        self.send_message(message)

    def send_message_sync_model_to_client(self, receive_id, global_model_params, client_index):
        message = Message(MyMessage.MSG_TYPE_S2C_SYNC_MODEL_TO_CLIENT, self.get_sender_id(), receive_id)
        message.add_params(MyMessage.MSG_ARG_KEY_MODEL_PARAMS, global_model_params)
        message.add_params(MyMessage.MSG_ARG_KEY_CLIENT_INDEX, str(client_index))
        message_size = message.get_message_size()
        self.benchmark.add_offline_bits(message_size)
        self.send_message(message)

    def check_whether_all_model_received(self):
        if len(self.active_clients) == len(self.clients_current_round) - self.dropout_num:
            return True
        return False

    def check_whether_all_masks_received(self):
        if len(self.dealer_client_map) == len(self.aggregated_masks):
            return True
        return False
