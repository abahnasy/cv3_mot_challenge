import time

def generate_time_stmp():
    return time.strftime('%Y-%m-%d-%H-%M-%S', time.gmtime())