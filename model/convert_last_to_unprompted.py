import argparse
import os
import ujson
from tqdm import tqdm


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        'Convert last beam to unprompted.'
    )

    # Configuration Parameters
    parser.add_argument('--split', default='validation')
    parser.add_argument('--dataset', default='xsum')
    parser.add_argument('--cand_dir', default='xsum_e_v1_ea')
    parser.add_argument('--unprompted_dir', default='xsum_ea_reg_1.0_0.1_0.1_unprompted')

    args = parser.parse_args()

    brio_dir = os.path.expanduser(os.path.join('~', 'BRIO', args.dataset))
    cand_dir = os.path.join(brio_dir, args.cand_dir, 'diverse', args.split)
    add_dir = os.path.join(brio_dir, args.unprompted_dir, 'diverse', args.split)
    out_dir = os.path.join(brio_dir, args.cand_dir + '_w_unprompted', 'diverse', args.split)

    print(f'Saving to {out_dir}')
    os.makedirs(out_dir, exist_ok=True)

    n = len(os.listdir(cand_dir))

    for idx in tqdm(range(n)):
        with open(os.path.join(add_dir, f'{idx}.json'), 'r') as fd:
            to_add = ujson.load(fd)

        with open(os.path.join(cand_dir, f'{idx}.json'), 'r') as fd:
            obj = ujson.load(fd)

        assert obj['article'] == to_add['article']

        assert len(to_add['candidates_untok']) == 1
        obj['candidates_untok'][-1] = to_add['candidates_untok'][0]
        assert len(to_add['candidates']) == 1
        obj['candidates'][-1] = to_add['candidates'][0]

        with open(os.path.join(out_dir, f'{idx}.json'), 'w') as fd:
            ujson.dump(obj, fd)
