"""Validate generated data structure, split separation, ID coverage and nesting."""
from pathlib import Path
import argparse
import json
import numpy as np

def validate(root):
    checks = 0
    expected = {'source': [3775, 783, 799], 'hour48': [3722, 773, 794], 'trio': [1883, 404, 404]}
    for split in ['primary', 'robust']:
        base = root / 'full' / split / 'data'
        meta = json.loads((base / 'meta.json').read_text())
        assert meta['n_entities'] == 63
        for task in expected:
            arrays = [dict(np.load(base / f'{task}_{part}.npz')) for part in ['fit', 'val', 'test']]
            sets = [set(z['cluster']) for z in arrays]
            assert all(not sets[i] & sets[j] for i in range(3) for j in range(i + 1, 3))
            known = set(arrays[0]['ids'].flat)
            assert all(set(z['ids'].flat) <= known for z in arrays[1:])
            if split == 'primary':
                assert [len(z['y']) for z in arrays] == expected[task]
            else:
                original = dict(np.load(root / 'full/primary/data' / f'{task}_test.npz'))
                for key in ['ids', 'y', 'row_id', 'cluster', 'focal']:
                    np.testing.assert_array_equal(arrays[2][key], original[key])
            checks += 1
    base = root / 'full/primary/data'
    fit = dict(np.load(base / 'hour48_fit.npz'))
    val = dict(np.load(base / 'hour48_val.npz'))
    test = dict(np.load(base / 'hour48_test.npz'))
    items = json.loads((root / 'fractions/subsets.json').read_text())
    assert len(items) == 31
    previous = {}
    for item in items:
        folder = root / 'fractions/subsets' / item['name'] / 'data'
        ix = np.asarray(item['indices'])
        z = dict(np.load(folder / 'hour48_fit.npz'))
        for key in fit:
            np.testing.assert_array_equal(z[key], fit[key][ix])
        for part, reference in [('val', val), ('test', test)]:
            with np.load(folder / f'hour48_{part}.npz') as current:
                for key in reference:
                    np.testing.assert_array_equal(current[key], reference[key])
        present = set(ix.tolist())
        assert previous.get(item['repeat'], set()) <= present
        previous[item['repeat']] = present
        assert len(ix) >= len(fit['y']) * item['pct'] / 100
        assert len(set(z['cluster'])) == item['clusters']
        assert set(z['ids'].flat) == set(item['ids_seen'])
        checks += 1
    print(f'Passed {checks} task/subset checks: split separation, coverage, fixed evaluation arrays and nested subsets.')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path(__file__).resolve().parent)
    validate(parser.parse_args().root.resolve())
