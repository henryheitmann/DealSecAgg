import json
import sys
from collections import OrderedDict

import numpy as np


class Message(object):

    MSG_ARG_KEY_OPERATION = "operation"
    MSG_ARG_KEY_TYPE = "msg_type"
    MSG_ARG_KEY_SENDER = "sender"
    MSG_ARG_KEY_RECEIVER = "receiver"

    MSG_OPERATION_SEND = "send"
    MSG_OPERATION_RECEIVE = "receive"
    MSG_OPERATION_BROADCAST = "broadcast"
    MSG_OPERATION_REDUCE = "reduce"

    MSG_ARG_KEY_MODEL_PARAMS = "model_params"

    def __init__(self, type=0, sender_id=0, receiver_id=0):
        self.type = type
        self.sender_id = sender_id
        self.receiver_id = receiver_id
        self.msg_params = {}
        self.msg_params[Message.MSG_ARG_KEY_TYPE] = type
        self.msg_params[Message.MSG_ARG_KEY_SENDER] = sender_id
        self.msg_params[Message.MSG_ARG_KEY_RECEIVER] = receiver_id

    def init(self, msg_params):
        self.msg_params = msg_params

    def init_from_json_string(self, json_string):
        self.msg_params = json.loads(json_string)
        self.type = self.msg_params[Message.MSG_ARG_KEY_TYPE]
        self.sender_id = self.msg_params[Message.MSG_ARG_KEY_SENDER]
        self.receiver_id = self.msg_params[Message.MSG_ARG_KEY_RECEIVER]
        # print("msg_params = " + str(self.msg_params))

    def get_sender_id(self):
        return self.sender_id

    def get_receiver_id(self):
        return self.receiver_id

    def add_params(self, key, value):
        self.msg_params[key] = value

    def get_params(self):
        return self.msg_params

    def add(self, key, value):
        self.msg_params[key] = value

    def get(self, key):
        return self.msg_params[key]

    def get_type(self):
        return self.msg_params[Message.MSG_ARG_KEY_TYPE]

    def to_string(self):
        return self.msg_params

    def to_json(self):
        json_string = json.dumps(self.msg_params)
        print("json string size = " + str(sys.getsizeof(json_string)))
        return json_string

    def get_content(self):
        print_dict = self.msg_params.copy()
        msg_str = str(self.__to_msg_type_string()) + ": " + str(print_dict)
        return msg_str

    def __to_msg_type_string(self):
        type = self.msg_params[Message.MSG_ARG_KEY_TYPE]
        return type

# Author: Henry Heitmann
    def get_message_size(self):
        total_size = 0
        for key in self.msg_params.keys():
            total_size += self.get_size(self.msg_params[key])
        return total_size

    def get_size(self, obj):
        if isinstance(obj, (dict, OrderedDict)):
            return sum([self.get_size(i) for i in obj.values()])
        if isinstance(obj, list):
            return sum([self.get_size(i) for i in obj])
        if isinstance(obj, np.ndarray):
            return obj.nbytes * 8
        else:
            return obj.__sizeof__() * 8
