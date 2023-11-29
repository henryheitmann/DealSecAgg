# Author: Henry Heitmann
import csv
import logging
import os
import time


class Benchmark:
    def __init__(self):
        self.round = 0
        self.start_time = self.current_time_millis()
        self.last_time = self.current_time_millis()

        self.offline_time = []
        self.training_time = []
        self.masking_time = []
        self.unmasking_time = []
        self.total_time = []

        self.offline_bits = []
        self.masking_bits = []
        self.unmasking_bits = []
        self.total_bits = []

    def current_time_millis(self):
        return round(time.time() * 1000)

    def set_start_time(self):
        self.start_time = self.current_time_millis()
        self.last_time = self.start_time

    def set_last_time(self):
        self.last_time = self.current_time_millis()

    def add_offline_time(self):
        current_time = self.current_time_millis()
        if len(self.offline_time) == self.round:
            self.offline_time.append(current_time - self.last_time)
        elif len(self.offline_time) == self.round + 1:
            self.offline_time[self.round] += (current_time - self.last_time)
        self.last_time = current_time

    def add_training_time(self):
        current_time = self.current_time_millis()
        if len(self.training_time) == self.round:
            self.training_time.append(current_time - self.last_time)
        elif len(self.training_time) == self.round + 1:
            self.training_time[self.round] += (current_time - self.last_time)
        self.last_time = current_time

    def add_masking_time(self):
        current_time = self.current_time_millis()
        if len(self.masking_time) == self.round:
            self.masking_time.append(current_time - self.last_time)
        elif len(self.masking_time) == self.round + 1:
            self.masking_time[self.round] += (current_time - self.last_time)
        self.last_time = current_time

    def add_unmasking_time(self):
        current_time = self.current_time_millis()
        if len(self.unmasking_time) == self.round:
            self.unmasking_time.append(current_time - self.last_time)
        elif len(self.unmasking_time) == self.round + 1:
            self.unmasking_time[self.round] += (current_time - self.last_time)
        self.last_time = current_time

    def set_total_time(self):
        current_time = self.current_time_millis()
        self.total_time.append(current_time - self.start_time)

    def add_offline_bits(self, bits):
        if len(self.offline_bits) == self.round:
            self.offline_bits.append(bits)
        elif len(self.offline_bits) == self.round + 1:
            self.offline_bits[self.round] += bits

    def add_masking_bits(self, bits):
        if len(self.masking_bits) == self.round:
            self.masking_bits.append(bits)
        elif len(self.masking_bits) == self.round + 1:
            self.masking_bits[self.round] += bits

    def add_unmasking_bits(self, bits):
        if len(self.unmasking_bits) == self.round:
            self.unmasking_bits.append(bits)
        elif len(self.unmasking_bits) == self.round + 1:
            self.unmasking_bits[self.round] += bits

    def set_total_bits(self, bits=0):
        total_bits = 0
        if len(self.offline_bits) > 0:
            total_bits += self.offline_bits[self.round]
        if len(self.masking_bits) > 0:
            total_bits += self.masking_bits[self.round]
        if len(self.unmasking_bits) > 0:
            total_bits += self.unmasking_bits[self.round]
        self.total_bits.append(total_bits + bits)

    def write_benchmark(self, path='test', filename='1.csv'):
        fields = []
        current_row = []
        if len(self.offline_time) > 0:
            fields.append('offline_time')
            current_row.append(self.offline_time[self.round])
        if len(self.training_time) > 0:
            fields.append('training_time')
            current_row.append(self.training_time[self.round])
        if len(self.masking_time) > 0:
            fields.append('masking_time')
            current_row.append(self.masking_time[self.round])
        if len(self.unmasking_time) > 0:
            fields.append('unmasking_time')
            current_row.append(self.unmasking_time[self.round])
        if len(self.total_time) > 0:
            fields.append('total_time')
            current_row.append(self.total_time[self.round])
        if len(self.offline_bits) > 0:
            fields.append('offline_bits')
            current_row.append(self.offline_bits[self.round])
        if len(self.masking_bits) > 0:
            fields.append('masking_bits')
            current_row.append(self.masking_bits[self.round])
        if len(self.unmasking_bits) > 0:
            fields.append('unmasking_bits')
            current_row.append(self.unmasking_bits[self.round])
        if len(self.total_bits) > 0:
            fields.append('total_bits')
            current_row.append(self.total_bits[self.round])

#        logging.info('Fields: {}'.format(fields))
#        logging.info('Benchmark: {}'.format(current_row))

        pathname = './benchmark/' + path

        if not os.path.exists(pathname):
            try:
                os.makedirs(pathname)
            except:
                logging.info("Path already exists")

        count = 0
        if os.path.exists(pathname + '/' + filename):
            with open(pathname + '/' + filename, 'r') as f:
                for count, line in enumerate(f):
                    count += 1

        if count == 0:
            with open(pathname + '/' + filename, 'w') as csvfile:
                csvwriter = csv.writer(csvfile)
                csvwriter.writerow(fields)
                csvwriter.writerow(current_row)
        else:
            with open(pathname + '/' + filename, 'a') as csvfile:
                csvwriter = csv.writer(csvfile)
                csvwriter.writerow(current_row)
        self.round += 1
