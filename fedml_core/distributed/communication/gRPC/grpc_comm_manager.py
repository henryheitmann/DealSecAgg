import logging
import os
import pickle
import threading
from concurrent import futures
from typing import List

import grpc

from ..gRPC import grpc_comm_manager_pb2_grpc, grpc_comm_manager_pb2

lock = threading.Lock()

from ...communication.base_com_manager import BaseCommunicationManager
from ...communication.message import Message
from ...communication.observer import Observer
from ...communication.gRPC.grpc_server import GRPCCOMMServicer
from ...communication.utils import log_communication_tick

import csv


class GRPCCommManager(BaseCommunicationManager):
    def __init__(self, host, port, ip_config_path, topic="fedml", client_id=0, client_num=0):
        # host is the ip address of server
        self.host = host
        self.port = str(port)
        self._topic = topic
        self.client_id = client_id
        self.client_num = client_num
        self._observers: List[Observer] = []

        if client_id == 0:
            self.node_type = "server"
        else:
            self.node_type = "client"
        self.opts = [
            ("grpc.max_send_message_length", 1000 * 1024 * 1024),
            ("grpc.max_receive_message_length", 1000 * 1024 * 1024),
        ]
        self.grpc_server = grpc.server(futures.ThreadPoolExecutor(max_workers=client_num), options=self.opts)
        self.grpc_servicer = GRPCCOMMServicer(host, port, client_num, client_id)
        grpc_comm_manager_pb2_grpc.add_gRPCCommManagerServicer_to_server(self.grpc_servicer, self.grpc_server)
        logging.info(os.getcwd())
        logging.info("&&&&&&&&&&&&&&& " + ip_config_path)
        self.ip_config = self._build_ip_table(ip_config_path)

        # starts a grpc_server on local machine using ip address "0.0.0.0"
        # host = self.ip_config[str(self.client_id)]
        # host = "127.0.0.1"
        self.grpc_server.add_insecure_port("{}:{}".format(host, port))
        logging.info("{}:{}".format(host, port))

        self.grpc_server.start()
        self.is_running = True
        print("server started. Listening on {}:{}".format(host, port))

    def send_message(self, msg: Message):
        logging.info("sending message to {}".format(msg))

        receiver_id = msg.get_receiver_id()

        log_communication_tick(self.client_id, receiver_id)

        logging.info("pickle.dumps(msg) START")
        msg_pkl = pickle.dumps(msg)
        # payload = msg.to_json()
        logging.info("pickle.dumps(msg) END")

        PORT_BASE = 50000
        # lookup ip of receiver from self.ip_config table
        receiver_ip = self.ip_config[str(receiver_id)]
        channel_url = "{}:{}".format(receiver_ip, str(PORT_BASE + receiver_id))

        channel = grpc.insecure_channel(channel_url, options=self.opts)
        stub = grpc_comm_manager_pb2_grpc.gRPCCommManagerStub(channel)

        request = grpc_comm_manager_pb2.CommRequest()

        request.client_id = self.client_id

        request.message = msg_pkl

        stub.sendMessage(request)
        logging.debug("sent successfully")
        channel.close()

    def add_observer(self, observer: Observer):
        self._observers.append(observer)

    def remove_observer(self, observer: Observer):
        self._observers.remove(observer)

    def handle_receive_message(self):
        thread = threading.Thread(target=self.message_handling_subroutine)
        thread.start()

    def message_handling_subroutine(self):
        while self.is_running:
            if self.grpc_servicer.message_q.qsize() > 0:
                lock.acquire()
                msg_pkl = self.grpc_servicer.message_q.get()
                logging.info("unpickle START")
                msg = pickle.loads(msg_pkl)
                logging.info("unpickle END")

                # logging.info("msg_params_string = {}".format(msg_params_string))
                # msg_params = Message()
                # msg_params.init_from_json_string(msg_params_string)
                logging.info("msg = {}".format(msg))
                msg_type = msg.get_type()
                for observer in self._observers:
                    observer.receive_message(msg_type, msg)
                lock.release()
        return

    def stop_receive_message(self):
        self.grpc_server.stop(None)
        self.is_running = False

    def notify(self, message: Message):
        msg_type = message.get_type()
        for observer in self._observers:
            observer.receive_message(msg_type, message)

    def _build_ip_table(self, path):
        ip_config = dict()
        with open(path, newline="") as csv_file:
            csv_reader = csv.reader(csv_file)
            # skip header line
            next(csv_reader)

            for row in csv_reader:
                receiver_id, receiver_ip = row
                ip_config[receiver_id] = receiver_ip.strip()
        return ip_config
