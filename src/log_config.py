# log_config.py

import logging
from datetime import datetime

class MillisecondFormatter(logging.Formatter):
    def formatTime(self, record, datefmt=None):
        dt = datetime.fromtimestamp(record.created)
        if datefmt:
            s = dt.strftime(datefmt)
            return s[:-3]  # Cắt 3 số cuối micro giây để thành mili giây
        else:
            return super().formatTime(record, datefmt)
