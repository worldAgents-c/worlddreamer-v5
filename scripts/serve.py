# Copyright (C) 2026 World Agents Corporation.
import argparse
import json
import os
from pathlib import Path
import sys
import tempfile

from worlddreamer_v5.utils.checkpoint import file_sha256
from worlddreamer_v5.utils.data_paths import data_path_mapping, remap_data_paths
from worlddreamer_v5.utils.model_package import package_model_config


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--cache', type=Path, required=True)
    parser.add_argument('--calibration', type=Path, required=True)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--host', default='127.0.0.1')
    parser.add_argument('--port', type=int, default=10100)
    parser.add_argument('--attention-backend', choices=('sdpa', 'flash_attention_2'), default='sdpa')
    parser.add_argument('--kernel-backend', choices=('torch', 'liger'), default='torch')
    args = parser.parse_args()
    os.environ['WORLDDREAMER_V5_KERNEL_BACKEND'] = args.kernel_backend
    bundle = args.checkpoint.resolve()
    metadata = json.loads((bundle / 'checkpoint.json').read_text())
    contract = metadata['robot_action_contract']
    cache = remap_data_paths(json.loads(args.cache.read_text()),
                data_path_mapping({'video_cache_manifest': args.cache}))
    model = package_model_config(bundle)
    model.update(attention_backend=args.attention_backend, async_train=False,
                 ffn_gradient_checkpointing=False)
    data = {key: contract[key] for key in ('state_dim', 'action_dim', 'action_length',
            'action_target', 'normalization_dataset', 'video_observed_latents',
            'video_future_latents')}
    data.update(normalize_path=str(bundle / 'normalize.json'),
                processor_path=str(bundle / 'assets/vlm'))
    config = {'model': model, 'data': data, 'conditioning': {
        'prompt_cache': cache['prompt_cache'],
        'prompt_cache_sha256': file_sha256(cache['prompt_cache']),
        'prompt_keys': cache['prompt_keys'], 'temporal_compression': cache['temporal_compression']},
        'inference': {'checkpoint': str(bundle), 'device': args.device,
         'vae_path': str(bundle / 'assets/video/vae'),
         'calibration': str(args.calibration.resolve()), 'host': args.host, 'port': args.port}}
    from worlddreamer_v5.server.inference import main as serve
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / 'runtime.json'
        path.write_text(json.dumps(config))
        sys.argv = [sys.argv[0], '--config', str(path)]
        serve()


if __name__ == '__main__':
    main()
