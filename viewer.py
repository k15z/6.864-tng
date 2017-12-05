# Draw plots of the logs
import math
import json
import pandas as pd
from glob import glob
from os import system

def parse_logfile(logfile):
    log = []
    with open(logfile, "rt") as fin:
        for line in fin:
            try:
                log.append(json.loads(line))
            except: pass
    return log

def go():
    system("rm logs/*.png")
    for logfile in glob("logs/*.log"):
        log = parse_logfile(logfile)
        if len(log) > 0:
            df = pd.DataFrame(log)
            df = df.loc[df['mode'] == "dev"].reset_index()
            ax = df.plot(x="index", ylim=(0.0, 1.0), title=logfile.replace(".log", ""))
            fig = ax.get_figure()
            fig.savefig(logfile.replace(".log", ".png"))

go()
