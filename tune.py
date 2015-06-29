#!/usr/bin/env python
"""tune.py
Usage: tune.py <config> [--steps=<STEPS>] [--concurrent=<NUM>]

Options:
    --steps=<STEPS>     how many random options to explore  [default: 100]
    --concurrent=<NUM>  how many parallel jobs to run   [default: 10]

"""

from docopt import docopt
import json
import itertools
import numpy as np
import subprocess
import os
from time import sleep

if __name__ == '__main__':
    args = docopt(__doc__)
    with open(args['<config>']) as f:
        config = json.load(f)

    program = config['program']

    list_of_options = []

    keys = []
    for key, option in config.items():
        if not key.startswith('--'):
            continue # not an option
        options = [key + '=' + str(op) for op in option]
        list_of_options.append(options)
        keys.append(key)

    print 'tuning', keys
    grid = [g for g in itertools.product(*list_of_options)]
    print 'generated', len(grid), 'combinations from', len(keys), 'options'

    def launch(program, options):
        folder = program.replace(' ', '_').replace('/', '_').replace('-', '')
        if not os.path.isdir(folder):
            os.makedirs(folder)
        filename = '_'.join(options).replace(' ', '_').replace('/', '_').replace('-', '')
        file = open(os.path.join(folder, filename + '.out'), 'wb')
        cmd = [program] + list(options)
        print ' '.join(cmd)
        process = subprocess.Popen(cmd, shell=True, stdout=file)
        return [file, process]

    steps, max_p = [int(args[k]) for k in ['--steps', '--concurrent']]
    np.random.shuffle(grid)
    handles = {}
    for step, options in enumerate(grid[:steps]):
        file, process = launch(program, options)
        handles[file] = process

        while len(handles) > max_p:
            for file, process in handles.items():
                if process.poll() is not None: # terminated
                    print 'finished', file.name
                    file.close()
                    del handles[file]
                    break
            sleep(20)
