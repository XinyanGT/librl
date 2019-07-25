import argparse, copy, importlib
import multiprocessing as mp
import itertools
from rl.core.utils.misc_utils import zipsame

def get_combs_and_keys(ranges):
    keys = []
    values = []
    for r in ranges:
        keys += r[::2]
    values = [list(zipsame(*r[1::2])) for r in ranges]
    cs = itertools.product(*values)
    combs = []
    for c in cs:
        comb = []
        for x in c:
            comb += x
        # print(comb)
        combs.append(comb)
    return combs, keys


def main(script_name, range_names, n_processes=-1, config_name=None):
    """ Run the `main` function in script_name.py in parallel with
        `n_processes` with different configurations given by `range_names`.

        Each configuration is jointly specificed by `config_name` and
        `range_names`. If `config_name` is None, it defaults to use the
        `CONFIG` dict in the script file. A valid config is a dict and must
        contains a key 'exp_name' whose value will be used to create the
        indentifier string to log the experiments.

        `range_names` is a list of string, which correspond to a range that
        specifies a set of parameters in the config dict one wish to experiment
        with. For example, if a string "name" is in `range_names`, the dict
        named `range_name` in script_name_ranges.py will be loaded. If
        script_name_ranges.py does not exist, it loads ranges.py.  The values
        of these experimental parameters will be used, jointly with `exp_name`,
        to create the identifier in logging.
    """
    # Set to the number of workers.
    # It defaults to the cpu count of your machine.
    if n_processes == -1:
        n_processes = None
    print('# of CPU (threads): {}'.format(mp.cpu_count()))

    script = importlib.import_module('scripts.'+script_name)
    try:
        script_ranges = importlib.import_module('scripts.'+script_name+'_ranges')
    except:
        script_ranges = importlib.import_module('scripts.ranges')
    template = getattr(script, 'CONFIG' or config_name)
    # Create the configs for all the experiments.
    tps = []
    for range_name in range_names:
        ranges = getattr(script_ranges, 'range_'+range_name)
        combs, keys = get_combs_and_keys(ranges)
        print('Total number of combinations: {}'.format(len(combs)))
        for i, comb in enumerate(combs):
            tp = copy.deepcopy(template)
            # Generate a unique exp name based on the provided ranges.
            # The description string start from the the exp name.
            value_strs = [tp['exp_name']]
            for (value, key) in zip(comb, keys):
                entry = tp
                for k in key[:-1]:  # walk down the template tree
                    entry = entry[k]
                # Make sure the key is indeed included in the template,
                # so that we set the desired flag.
                assert key[-1] in entry
                entry[key[-1]] = value
                if key[-1]=='seed':
                    continue # do not include seed number
                else:
                    if value is True:
                        value = 'T'
                    if value is False:
                        value = 'F'
                    value_strs.append(str(value).split('/')[0])
            tp['exp_name'] = '-'.join(value_strs)
            tps.append(tp)

    # Launch the experiments.
    with mp.Pool(processes=n_processes, maxtasksperchild=1) as p:
        p.map(script.main, tps, chunksize=1)
        #p.map(func, tps, chunksize=1)


def func(tp):
    print(tp['exp_name'], tp['general']['seed'])


if __name__ == '__main__':
    mp.set_start_method('spawn')

    parser = argparse.ArgumentParser()
    parser.add_argument('script_name')
    parser.add_argument('-r', '--range_names', nargs='+')
    parser.add_argument('-c', '--config_name', type=str)
    parser.add_argument('--n_processes', type=int, default=-1)
    args = parser.parse_args()
    main(args.script_name, args.range_names,
         n_processes=args.n_processes, config_name=args.config_name)