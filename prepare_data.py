"""Download pinned upstream inputs and regenerate all model inputs, without training."""
from pathlib import Path
import argparse
import concurrent.futures
import hashlib
import json
import subprocess
import sys
import urllib.parse
import urllib.request

ROOT = Path(__file__).resolve().parent

def fetch(raw):
    manifest = json.loads((ROOT / 'preprocessing/source_manifest.json').read_text())
    def one(record):
        path = raw / record['file']
        path.parent.mkdir(parents=True, exist_ok=True)
        expected = record['sha256']
        if not path.exists():
            url = ('https://raw.githubusercontent.com/amichaibk/community_effects/'
                   + manifest['commit'] + '/' + urllib.parse.quote(record['file']))
            with urllib.request.urlopen(url, timeout=120) as response:
                content = response.read()
            if hashlib.sha256(content).hexdigest() != expected:
                raise ValueError(f"Source hash mismatch: {record['file']}")
            path.write_bytes(content)
        if hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            raise ValueError(f"Cached input hash mismatch: {record['file']}")
        return record['file']
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        files = list(pool.map(one, manifest['files']))
    print(f'Verified {len(files)} pinned upstream files.', flush=True)


def subsets(output):
    import copy
    import numpy as np
    source = output / 'full/primary/data'
    target = output / 'fractions'
    target.mkdir(parents=True, exist_ok=True)
    plan = json.loads((ROOT / 'fractions/plan.json').read_text())
    with np.load(source / 'hour48_fit.npz') as z:
        fit = dict(z)
    with np.load(source / 'hour48_val.npz') as z:
        val = dict(z)
    with np.load(source / 'hour48_test.npz') as z:
        test = dict(z)
    meta = json.loads((source / 'meta.json').read_text())
    keys = np.unique(fit['cluster'])
    items = []
    for repeat in range(10):
        seed = 2026100201 + repeat
        order = np.random.default_rng(seed).permutation(keys)
        counts = np.array([np.sum(fit['cluster'] == k) for k in order])
        cumulative = np.cumsum(counts)
        for pct in [10, 25, 50, 100]:
            if pct == 100 and repeat:
                continue
            stop = np.searchsorted(cumulative, len(fit['y']) * pct / 100, side='left') + 1
            indices = np.flatnonzero(np.isin(fit['cluster'], order[:stop]))
            data = {k: v[indices] for k, v in fit.items()}
            scale = (dict(meta['scales']['hour48']) if pct == 100 else
                     dict(mean=float(data['y'].mean()), std=float(data['y'].std())))
            name = f'q{pct:03d}_r{repeat:02d}'
            item = dict(name=name, pct=pct, repeat=repeat, sampling_seed=seed,
                        rows=len(indices), clusters=len(np.unique(data['cluster'])),
                        indices=indices.tolist(),
                        focal_counts={str(k): int(np.sum(data['focal'] == k)) for k in np.unique(data['focal'])},
                        ids_seen=np.unique(data['ids']).tolist(), scale=scale)
            items.append(item)
            dest = target / 'subsets' / name
            (dest / 'data').mkdir(parents=True, exist_ok=True)
            for part, z in [('fit', data), ('val', val), ('test', test)]:
                np.savez_compressed(dest / 'data' / f'hour48_{part}.npz', **z)
            local_meta = copy.deepcopy(meta)
            local_meta['scales'] = {'source': scale, 'hour48': scale}
            (dest / 'data/meta.json').write_text(json.dumps(local_meta, indent=2))
            (dest / 'plan.json').write_text(json.dumps(plan, indent=2))
    (target / 'subsets.json').write_text(json.dumps(items, indent=2))
    print(f'Generated {len(items)} nested training subsets.', flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-root', type=Path, default=ROOT,
                        help='Destination for generated full/ and fractions/ data.')
    parser.add_argument('--raw-dir', type=Path, default=ROOT / '.cache/upstream',
                        help='Download/cache directory; existing files are hash-checked.')
    args = parser.parse_args()
    output = args.output_root.resolve()
    raw = args.raw_dir.resolve()
    fetch(raw)
    pilot = output / '.cache/pilot'
    def run(script, *argv):
        subprocess.run([sys.executable, str(ROOT / 'preprocessing' / script),
                        *map(str, argv)], check=True)
    run('pilot.py', pilot, raw)
    for split in ['primary', 'robust']:
        run('full.py', output / 'full' / split, raw, pilot)
    run('measured_effects.py', output / 'full', raw)
    subsets(output)
    print('Input preparation complete. No models trained.', flush=True)

if __name__ == '__main__':
    main()
