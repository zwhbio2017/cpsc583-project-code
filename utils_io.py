# Supplementary function for MolecularMetrics 

import datetime
import string
import random


def get_date_postfix():
    dt = datetime.datetime.now()
    post_fix = '{}_{:02d}-{:02d}-{:02d}'.format(
        dt.date(), dt.hour, dt.minute, dt.second)

    return post_fix


def random_string(string_len=3):
    letters = string.ascii_lowercase
    return ''.join(random.choice(letters) for i in range(string_len))