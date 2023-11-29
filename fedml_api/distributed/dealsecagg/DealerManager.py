# Author: Henry Heitmann

import logging
from collections import OrderedDict

from fedml_api.distributed.utils.Benchmark import Benchmark
from fedml_api.distributed.utils.masking import generate_private_mask_from_shape, add_models
from fedml_core.distributed.communication.message import Message
from fedml_core.distributed.dealer.dealer_manager import DealerManager
from .message_define import MyMessage


class DealerManager(DealerManager):
    def __init__(self, args, comm=None, rank=0, size=0):
        super().__init__(args, comm, rank, size)
        self.benchmark = Benchmark()
        self.args = args
        self.round_num = args.comm_round

        self.seed_map = dict()

    def run(self):
        super().run()

    def register_message_receive_handlers(self):
        self.register_message_receive_handler(
            MyMessage.MSG_TYPE_C2D_SEND_MASK_TO_DEALER, self.handle_message_mask_from_client
        )
        self.register_message_receive_handler(
            MyMessage.MSG_TYPE_S2D_RECOVER_MASKS, self.handle_message_recover_masks
        )

    def handle_message_mask_from_client(self, msg_params):
        self.benchmark.set_last_time()
        sender_id = msg_params.get(MyMessage.MSG_ARG_KEY_SENDER)
        seed = msg_params.get(MyMessage.MSG_ARG_KEY_MASK)
        logging.info("Dealer %d received mask from Client %d" % (self.get_sender_id(), sender_id))

        self.seed_map[sender_id] = seed
        self.benchmark.add_masking_time()

    def handle_message_recover_masks(self, msg_params):
        self.benchmark.set_last_time()
        active_clients = msg_params.get(MyMessage.MSG_ARG_KEY_CLIENT_ID_LIST)
        model_shape = msg_params.get(MyMessage.MSG_ARG_KEY_MODEL_SHAPE)
        logging.info("Dealer %d aggregates mask" % self.get_sender_id())

        aggregated_mask = OrderedDict()
        datatype = self.args.quantization

        for idx, client in enumerate(active_clients):
            mask = generate_private_mask_from_shape(self.seed_map[client], model_shape, datatype)
            if idx == 0:
                aggregated_mask = mask
            else:
                aggregated_mask = add_models(aggregated_mask, mask)

        self.seed_map.clear()
        logging.info("Dealer %d has finished aggregation of masks" % self.get_sender_id())

        self.send_aggregated_mask_to_server(aggregated_mask)

        self.benchmark.add_unmasking_time()
        self.benchmark.set_total_bits()
        self.benchmark.set_total_time()
        self.benchmark.write_benchmark(
            path=self.args.model + '_' + str(self.args.client_num_in_total) + '_'
                 + str(self.args.client_num_per_round) + '_' + str(self.args.num_dropouts) + '_'
                 + self.args.quantization + '_' + str(self.args.total_num_dealers) + '_' + str(self.args.num_dealers),
            filename='dealer')

    def send_aggregated_mask_to_server(self, aggregated_mask):
        message = Message(MyMessage.MSG_TYPE_D2S_SEND_AGGREGATED_MASK_TO_SERVER, self.get_sender_id(), 0)
        message.add_params(MyMessage.MSG_ARG_KEY_AGGREGATED_MASKS, aggregated_mask)
        message_size = message.get_message_size()
        self.benchmark.add_unmasking_bits(message_size)
        self.send_message(message)
