import logging
import time

class CustomFormatter(logging.Formatter):
    def format(self, record):
        # Thêm thuộc tính epoch_ms vào record
        record.epoch_ms = int(time.time() * 1000)
        return super().format(record)