# Copyright (C) 2026 World Agents Corporation.
import argparse
import copy
import hashlib
import json
from pathlib import Path
import shutil

TASKS = ('PlaceFood', 'CameraAlignment', 'TakePhoto')


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024**2), b''):
            digest.update(block)
    return digest.hexdigest()


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + '\n')


def rewrite(value, mapping):
    if isinstance(value, str):
        return mapping.get(value, value)
    if isinstance(value, list):
        return [rewrite(item, mapping) for item in value]
    if isinstance(value, dict):
        return {key: rewrite(item, mapping) for key, item in value.items()}
    return value


def prepare(args):
    source = json.loads(args.manifest.read_text())
    cache = json.loads(args.cache_manifest.read_text())
    digest = sha256(args.manifest)
    if cache['source_manifest_sha256'] != digest:
        raise ValueError('source dataset and video cache are not paired')
    selected = [(i, r) for i, r in enumerate(source['records']) if r['task'] in TASKS]
    if {r['task'] for _, r in selected} != set(TASKS):
        raise ValueError('one or more release tasks are missing')
    profile = copy.deepcopy(json.loads(args.normalize.read_text())['datasets']['RoboFactory'])
    if profile['action_dim'] != 18 or profile['state_dim'] != 20:
        raise ValueError('normalization must use state20 and action18')
    if set(profile['tasks']) != set(TASKS):
        raise ValueError('normalization was computed on a different task selection')
    if digest not in [profile.get('source_manifest_sha256'), *profile.get('source_manifest_sha256s', [])]:
        raise ValueError('normalization is not bound to the input dataset')
    profile['units']['waist'] = 'm'
    args.output.mkdir(parents=True, exist_ok=False)
    inventory, records, cache_records = {}, [], {}
    mapping = {}

    def transfer(source_path, relative):
        source_path = Path(source_path)
        target = args.output / relative
        if not source_path.is_file():
            raise FileNotFoundError(source_path)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source_path, target)
        digest = sha256(target)
        if digest != sha256(source_path):
            raise IOError(f'copy checksum mismatch: {relative}')
        inventory[relative] = {'sha256': digest, 'size': target.stat().st_size}
        return '@data/' + relative

    for index, (old_index, record) in enumerate(selected):
        stem = f"episodes/{record['task']}/{record['episode']:06d}"
        for key, name in [('source_h5', 'trajectory.h5'), ('config', 'env_config.yaml')]:
            mapping[record[key]] = transfer(record[key], f'{stem}/{name}')
        for role, path in record['videos'].items():
            mapping[path] = transfer(path, f'{stem}/{role}.mp4')
        annotations = []
        for agent, path in enumerate(record['annotations']):
            original = json.loads(Path(path).read_text())
            portable = rewrite(original, mapping)
            portable['robofactory'] = {key: value for key, value in portable['robofactory'].items()
                if key in ('schema_version', 'task', 'episode', 'seed', 'active_source_agent',
                           'action_target', 'ever_success', 'final_success', 'camera_data',
                           'valid_action_frames', 'terminal_action', 'action_time_alignment',
                           'source_command_sha256', 'robot_urdf_sha256')}
            # All numeric labels, masks, instructions and camera keys are retained.
            for key in ('proprios', 'actions', 'instruction'):
                if original.get(key) != portable.get(key):
                    raise ValueError(f'annotation content changed: {key}')
            relative = f'{stem}/agent_{agent + 1}.json'
            write_json(args.output / relative, portable)
            inventory[relative] = {'sha256': sha256(args.output / relative),
                                   'source_sha256': sha256(path)}
            annotations.append('@data/' + relative)
        retained = {k: record[k] for k in ('task', 'episode', 'seed', 'frame_count',
                    'agent_count', 'valid_action_frames', 'action_target', 'legacy_numpy_seed',
                    'camera_variant') if k in record}
        retained.update(source_h5=mapping[record['source_h5']],
                        config=mapping[record['config']], annotations=annotations,
                        videos=rewrite(record['videos'], mapping))
        records.append(retained)
        entry = cache['records'][str(old_index)]
        relative = f'cache/records/{index:05d}.safetensors'
        transfer(entry['path'], relative)
        cache_records[str(index)] = {'path': '@cache/records/' + f'{index:05d}.safetensors',
                                    'task': record['task'], 'episode': record['episode']}
    transfer(source['calibration'], 'calibration.json')
    transfer(cache['prompt_cache'], 'cache/prompts.safetensors')
    manifest = {'state_dim': 20, 'action_dim': 18, 'action_target': source['action_target'],
                'tasks': list(TASKS), 'record_count': len(records), 'records': records}
    write_json(args.output / 'manifest.json', manifest)
    new_digest = sha256(args.output / 'manifest.json')
    profile.pop('source_manifest_sha256s', None)
    profile['source_manifest_sha256'] = new_digest
    write_json(args.output / 'normalize.json', {'format': 'worlddreamer-v5.normalization/v2',
               'datasets': {'RoboFactory': profile}})
    write_json(args.output / 'cache/manifest.json', {
        'source_manifest_sha256': new_digest, 'record_count': len(records),
        'temporal_compression': cache['temporal_compression'],
        'prompt_cache': '@cache/prompts.safetensors',
        'prompt_keys': {task: cache['prompt_keys'][task] for task in TASKS},
        'records': cache_records})
    write_json(args.output / 'integrity.json', {'source_manifest_sha256': digest,
               'source_cache_sha256': sha256(args.cache_manifest), 'files': inventory})
    print(json.dumps({'output': str(args.output), 'records': len(records),
                      'frames': sum(r['frame_count'] for r in records)}))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--cache-manifest', type=Path, required=True)
    parser.add_argument('--normalize', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    prepare(parser.parse_args())
