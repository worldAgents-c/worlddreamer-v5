# Copyright (C) 2026 World Agents Corporation.
import argparse
import json
from pathlib import Path
import shutil

from worlddreamer_v5.utils.checkpoint import file_sha256, load_tensor_source, validate_state
from worlddreamer_v5.models.variants import MODEL_NAME, normalize_model_config


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--template', type=Path, required=True)
    parser.add_argument('--normalize', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    from safetensors import safe_open
    from safetensors.torch import save_file
    state, checkpoint = load_tensor_source(args.checkpoint)
    validate_state(state)
    metadata = checkpoint.get('client_state', checkpoint)
    contract = metadata['robot_action_contract']
    if (contract['state_dim'], contract['action_dim'], contract['action_length']) != (20, 18, 30):
        raise ValueError('expected state20/action18 with 30-step chunks')
    if file_sha256(args.normalize) != contract['normalize_sha256']:
        raise ValueError('normalization differs from training')
    with safe_open(str(args.template / 'model.safetensors'), framework='pt') as template:
        if set(template.keys()) != set(state):
            raise ValueError('template and trained model have different tensor keys')
        for name, value in state.items():
            if tuple(template.get_slice(name).get_shape()) != tuple(value.shape):
                raise ValueError(f'template shape differs: {name}')
    args.output.mkdir(parents=True, exist_ok=False)
    shutil.copytree(args.template / 'assets', args.output / 'assets')
    model_config = json.loads((args.template / 'model_config.json').read_text())
    model_config['model'] = normalize_model_config(model_config['model'])
    (args.output / 'model_config.json').write_text(json.dumps(model_config, indent=2) + '\n')
    shutil.copyfile(args.normalize, args.output / 'normalize.json')
    save_file({k:v.detach().cpu().contiguous().clone() for k,v in state.items()},
              str(args.output / 'model.safetensors'))
    release = json.loads((args.template / 'checkpoint.json').read_text())
    release['model'] = MODEL_NAME
    release['robot_action_contract'] = contract
    release['model_sha256'] = file_sha256(args.output / 'model.safetensors')
    release['asset_sha256'] = {str(p.relative_to(args.output)): file_sha256(p)
         for p in args.output.rglob('*') if p.is_file() and p.name != 'model.safetensors'}
    (args.output / 'checkpoint.json').write_text(json.dumps(release, indent=2) + '\n')
    print(args.output)


if __name__ == '__main__':
    main()
