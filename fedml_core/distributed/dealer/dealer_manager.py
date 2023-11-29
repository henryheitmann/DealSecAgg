# Author: Henry Heitmann

import logging
from abc import abstractmethod

from mpi4py import MPI

from ..communication.mpi.com_manager import MpiCommunicationManager
from ..communication.observer import Observer


class DealerManager(Observer):

    def __init__(self, args, comm=None, rank=0, size=0):
        self.args = args
        self.size = size
        self.rank = rank

        self.backend = "MPI"
        self.com_manager = MpiCommunicationManager(comm, rank, size, node_type="dealer")
        self.com_manager.add_observer(self)
        self.message_handler_dict = dict()

    def run(self):
        self.register_message_receive_handlers()
        self.com_manager.handle_receive_message()
        print('done running')

    def get_sender_id(self):
        return self.rank

    def receive_message(self, msg_type, msg_params) -> None:
        handler_callback_func = self.message_handler_dict[msg_type]
        handler_callback_func(msg_params)

    def send_message(self, message):
        self.com_manager.send_message(message)

    @abstractmethod
    def register_message_receive_handlers(self) -> None:
        pass

    def register_message_receive_handler(self, msg_type, handler_callback_func):
        self.message_handler_dict[msg_type] = handler_callback_func

    def finish(self):
        logging.info("__finish dealer")
        MPI.COMM_WORLD.Abort()
