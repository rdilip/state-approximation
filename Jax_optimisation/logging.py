"""
Convenience functions for printing information to the console and/or to file
"""

import io
import time
from typing import Optional
from pathlib import Path


class Logger:
    def __init__(self, logfile, print_logs, prefix=''):
        if logfile is not None:
            if type(logfile) == str:
                logfile = Path(logfile).expanduser()
            logfile.parent.mkdir(parents=True, exist_ok=True)

        self.logfile = logfile
        self.print_logs = print_logs

        if len(prefix) > 0 and prefix[-1] != ' ':
            prefix += ' '
        self.prefix = prefix

    def log(self, msg, timestamp=False, prefix=None, file_only=False, print_only=False):
        msg = repr(msg)

        prefix = self.prefix if prefix is None else prefix
        if len(prefix) > 0 and prefix[-1] != ' ':
            prefix += ' '

        if timestamp is not False:
            if timestamp is True or timestamp in ['full']:
                msg = f'[{time.asctime()}]' + msg
            elif timestamp in ['short', 'clock']:
                msg = f'[{time.asctime()[11:19]}]' + msg
        msg = prefix + msg
        log(msg, None if print_only else self.logfile, self.print_logs and not file_only)

    def warn(self, msg, force_print=False):
        print_logs = force_print or self.print_logs
        log('[WARNING] ' + self.prefix + msg, self.logfile, print_logs)

    def vline(self, file_only=False, print_only=False):
        self.log('-' * 40, prefix=None, file_only=file_only, print_only=print_only)

    def lineskip(self, file_only=False, print_only=False):
        self.log('', prefix=None, file_only=file_only, print_only=print_only)

    def copy(self, prefix: Optional[str] = None):
        prefix = self.prefix if prefix is None else prefix
        return Logger(self.logfile, self.print_logs, prefix)


def log(msg, logfile, print_logs):
    if logfile is not None:
        with io.open(logfile, 'a', buffering=1, newline='\n') as lf:
            lf.write(msg + '\n')
    if print_logs:
        print(msg)
