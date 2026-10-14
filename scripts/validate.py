# Copyright (C) 2026 World Agents Corporation.
import argparse
import json
import os
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--data', type=Path, required=True)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--attention-backend', choices=('sdpa', 'flash_attention_2'), default='sdpa')
    parser.add_argument('--kernel-backend', choices=('torch', 'liger'), default='torch')
    parser.add_argument('--samples-per-task', type=int, default=1)
    parser.add_argument('--data-only', action='store_true')
    args = parser.parse_args()
    os.environ['WORLDDREAMER_V5_KERNEL_BACKEND'] = args.kernel_backend
    import torch
    from worlddreamer_v5.data.dataset import WorldDataset
    from worlddreamer_v5.data.collator import WorldCollator
    from worlddreamer_v5.utils.contract import data_contract
    from worlddreamer_v5.utils.config import build_model
    from worlddreamer_v5.utils.model_package import package_model_config
    from worlddreamer_v5.utils.checkpoint import load_model_weights, file_sha256
    from worlddreamer_v5.utils.normalization_registry import load_statistics
    from worlddreamer_v5.utils.normalization import validate_stats, validate_quantiles
    params = {'manifest_path': str(args.data / 'manifest.json'),
              'normalize_path': str(args.data / 'normalize.json'),
              'video_cache_manifest': str(args.data / 'cache/manifest.json'),
              'state_dim': 20, 'action_dim': 18, 'action_length': 30,
              'action_target': 'commanded_tcp_pose_and_gripper_drive_v3',
              'normalization_dataset': 'RoboFactory', 'image_augmentation': False}
    data_contract(params)
    dataset = WorldDataset(params)
    metadata = json.loads((args.checkpoint / 'checkpoint.json').read_text())
    contract = metadata['robot_action_contract']
    normalization_path = args.checkpoint / 'normalize.json'
    if file_sha256(normalization_path) != contract['normalize_sha256']:
        raise ValueError('checkpoint normalization checksum mismatch')
    for key in ('state_dim', 'action_dim', 'action_length', 'action_target', 'normalization_dataset'):
        if contract[key] != params[key]:
            raise ValueError(f'checkpoint and recorded data disagree: {key}')
    stats = load_statistics(normalization_path, params['normalization_dataset'])
    if stats['action_target'] != params['action_target']:
        raise ValueError('checkpoint normalization action semantics mismatch')
    if not set(dataset.task_indices).issubset(stats['tasks']):
        raise ValueError('recorded tasks are outside checkpoint normalization')
    dataset.mean, dataset.std = validate_stats(stats['mean'], stats['std'], dataset.action_length)
    dataset.q01, dataset.q99 = validate_quantiles(stats['q01'], stats['q99'])
    collate = WorldCollator(str(args.checkpoint / 'assets/vlm'))
    model = None
    if not args.data_only:
        config = package_model_config(args.checkpoint)
        config.update(async_train=False, attention_backend=args.attention_backend,
                      ffn_gradient_checkpointing=False)
        model = build_model(config)
        load_model_weights(model, args.checkpoint)
        model.eval().to(args.device)
    reports = []
    for task, indices in sorted(dataset.task_indices.items()):
        chosen = sorted(set(indices[int(i * (len(indices)-1) / max(args.samples_per_task-1, 1))]
                            for i in range(args.samples_per_task)))
        for index in chosen:
            batch = collate([dataset[index]])
            entry = {'task': task, 'index': index, 'state': list(batch['state'].shape),
                     'action': list(batch['action'].shape), 'finite': True}
            for name, value in batch.items():
                if isinstance(value, torch.Tensor) and not torch.isfinite(value).all():
                    raise ValueError(f'nonfinite batch: {name}')
            if model is not None:
                batch = {k: v.to(args.device) if isinstance(v, torch.Tensor) else v
                         for k,v in batch.items()}
                batch['video_latents'] = batch['video_latents'][:,:,:,:3]
                batch['video_latent_valid'] = batch['video_latent_valid'][:,:3]
                torch.manual_seed(42 + index)
                with torch.inference_mode():
                    prediction = model.generate(batch)
                assert prediction.shape == batch['action'].shape
                assert torch.isfinite(prediction).all()
                mask = batch['action_mask'].bool()
                entry['normalized_action_mse'] = float((prediction.float()-batch['action'].float())[mask].square().mean())
            reports.append(entry)
            print(json.dumps(entry), flush=True)
    print(json.dumps({'samples': len(reports), 'data_only': args.data_only}))


if __name__ == '__main__':
    main()
