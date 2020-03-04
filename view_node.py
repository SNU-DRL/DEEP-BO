import os
import sys
import subprocess
import argparse

try:
    from pathlib import Path
    import hiplot as hip
except:
	raise ImportError("In order to monitor the optimization progress, install hiplot on python 3.5 or above.") 

DEFAULT_PORT = 5005

def choose_csv(path, name):
    f_list = os.listdir(path)
    csv_fs = [f for f in f_list if f.endswith(".csv")]
    cands = []
    for c in csv_fs:
        if name == c:
            return "{}{}".format(path, c)
        elif name in c:
            cands.append(c)
    if len(cands) == 0:
        return None
    else:
        # get the latest modified file name
        mts = []
        for c in cands:
            mt = os.path.getmtime("{}{}".format(path, c))
            mts.append(mt)
        max_mt = max(mts)
        max_index = mts.index(max_mt)
        return "{}{}".format(path, cands[max_index])

def fetch_experiment(uri):
    # Only apply this fetcher if the URI starts with myxp://
    PREFIX = "myxp://"
    SAVE_DIR = "temp/"
    if not uri.startswith(PREFIX):
        # Let other fetchers handle this one
        csv_f = choose_csv(SAVE_DIR, uri)
    else:
        csv_f = choose_csv(SAVE_DIR, uri[len(PREFIX):])

    if csv_f != None:

        print("Loading experiments from {}".format(csv_f))
        return hip.Experiment.from_csv(csv_f)
    else:
        print("No uri: {}".format(uri))
        raise hip.ExperimentFetcherDoesntApply()            

def run_server(port):
    try:
        args = 'hiplot --host 0.0.0.0 --port {} view_node.fetch_experiment'.format(port)        
        subprocess.call(args, shell=True)
        pass
    except KeyboardInterrupt as ki:
        print("Server will be terminated by the kill request.")
        sys.exit(-1)
    except Exception as ex:
        print(ex)         

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('-p', '--port', default=DEFAULT_PORT, type=int,
                        help='The port number of the view node. The default is {}.'.format(DEFAULT_PORT))
    args = parser.parse_args()
    run_server(args.port)

