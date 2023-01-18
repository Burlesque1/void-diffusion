requirements = [ 'transformers' ]
import time
def run(args, callback):
    while True:
        print (args['seeds'][0])
        time.sleep(1)
        callback(0, "Waiting...")
        time.sleep(1)
        callback(50, "Running...")
        time.sleep(1)
        callback(100, "Complete!")
    return True