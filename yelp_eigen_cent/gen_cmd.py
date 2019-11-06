from Config import Config

import os
import shutil, argparse
from time import time
import importlib
import numpy as np

parser = argparse.ArgumentParser(description="training options")

parser.add_argument('--gen', action='store', dest='gen', required=True)
parser.add_argument('--restore-ckpt', action='store_true', dest='restore_ckpt', default=False)

args = parser.parse_args()

gen_path = os.path.relpath(args.gen)
gen_name = gen_path.replace('/', '.')

gen_cmd_setting = importlib.import_module(gen_name)

saving_path = gen_cmd_setting.save_path
tmplt_path = gen_cmd_setting.tmplt_path
cmdstr = gen_cmd_setting.cmdstr
cmdstr_restore = gen_cmd_setting.cmdstr_restore

settings = gen_cmd_setting.settings

configs = [o.items()for o in settings]
attr_len = [len(o) for o in configs]

def gen_comb(ref, entry):
    collect_comb = []
    if entry >= len(ref):
        return collect_comb
    ref = list(ref)
    attr = ref[entry][0]
    for val in ref[entry][1]:
        ret = gen_comb(ref, entry+1)
        if len(ret)>0:
            collect_comb += [o + [(attr, val)] for o in ret]
        else:
            collect_comb += [[(attr, val)]]
    return collect_comb

def valid_entry(save_path):

    def read_status(status_path):
        if not os.path.exists(status_path):
            return 'error'
        fd = open(status_path, 'r')
        time_stamp = float(fd.read().strip())
        fd.close()
        if time_stamp < 10.:
            return 'finished'
        cur_time = time()
        if cur_time - time_stamp < 1000.:
            return 'running'
        else:
            return 'error'


    if not os.path.exists(save_path):
        return False
    if read_status(save_path+'/status') == 'running':
        return True
    if read_status(save_path+'/status') == 'finished':
        return True
    if read_status(save_path+ '/status') == 'error':
        return False

    raise ValueError('unknown error')


def gen_cmd(comb):
    for item in comb:
        config = Config()
        config.loadConfig(tmplt_path)
        config_name = ''
        for (attr, val) in item:
            getattr(config, attr)
            setattr(config, attr, val)
            if isinstance(val, list):
                val = ','.join([str(o) for o in val])
            config_name+=attr+'-{}_'.format(val)

        save_path = saving_path+'/'+config_name

        if not valid_entry(save_path):
            if not args.restore_ckpt:
                if os.path.exists(save_path) and os.path.exists(save_path+'/checkpoint'):
                    raise ValueError('%s already exits. \n If want to recover from a '
                                     'checkpoint please specify --restore-ckpt' % save_path)
                if not os.path.exists(save_path):
                    os.mkdir(save_path)
                config.saveConfig(save_path+'/config')
                print(cmdstr.format(save_path, save_path))
            else:
                print(cmdstr_restore.format(save_path, save_path))

combs = [gen_comb(o, 0) for o in configs]

while len(combs)>0:
    idx = np.random.randint(0, len(combs))
    gen_cmd(combs.pop(idx))
