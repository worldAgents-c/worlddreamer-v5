# Copyright (C) 2026 World Agents Corporation.
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
import tempfile


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument(
        "--resume",
        help="worlddreamer-v5 Lightning/DeepSpeed training checkpoint, including optimizer state",
    )
    args = parser.parse_args()
    from lightning import Trainer, seed_everything
    from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint
    from lightning.pytorch.loggers import CSVLogger
    from lightning.pytorch.strategies import DeepSpeedStrategy
    from worlddreamer_v5.data.datamodule import WorldDataModule
    from worlddreamer_v5.models.runner.contract_guard import TrainingContractGuard
    from worlddreamer_v5.models.runner.worldDreamerv5_runner import WorldDreamerv5Runner

    from worlddreamer_v5.utils.config import load_config

    config = load_config(args.config)
    from worlddreamer_v5.utils.model_package import package_model_config
    from worlddreamer_v5.utils.normalization_registry import load_statistics
    import numpy as np
    bundle = Path(config['initialization']['checkpoint']).resolve()
    architecture = package_model_config(bundle)
    architecture.update(config['model'])
    config['model'] = architecture
    config['data']['processor_path'] = str(bundle / 'assets/vlm')
    source_stats = load_statistics(bundle / 'normalize.json', 'RoboFactory')
    data_stats = load_statistics(config['data']['normalize_path'], 'RoboFactory')
    for key in ('mean', 'std', 'q01', 'q99'):
        if not np.array_equal(source_stats[key], data_stats[key]):
            raise ValueError(f'training statistics differ from initialization: {key}')
    trainer_options = dict(config["trainer"])
    root = Path(trainer_options["default_root_dir"])
    root.mkdir(parents=True, exist_ok=True)
    # The resolved config is an audit artifact and may not overwrite another run.
    resolved = root / "worlddreamer_v5_config.json"
    text = json.dumps(config, indent=2) + "\n"
    with tempfile.NamedTemporaryFile(
        mode="w", dir=root, prefix=".config-", delete=False
    ) as handle:
        handle.write(text)
        handle.flush()
        os.fsync(handle.fileno())
        temporary = Path(handle.name)
    try:
        # Publish a complete file atomically, including concurrent DDP workers.
        try:
            os.link(temporary, resolved)
        except FileExistsError:
            if resolved.read_text() != text:
                raise ValueError(
                    "output directory already belongs to a different configuration"
                )
    finally:
        temporary.unlink()
    seed_everything(config.get("seed", 42), workers=True)
    guard_options = dict(config.get("guard", {}))
    guard = TrainingContractGuard(config["data"], str(root), **guard_options)
    data = WorldDataModule(config["data"])
    initialization = dict(config["initialization"])
    runner = WorldDreamerv5Runner(
        config["model"],
        config["optimizer"],
        config["scheduler"],
        expected_contract=guard.contract,
        **initialization,
    )
    if args.resume:
        # Resume format is checked by the module callback during restore.
        runner.resume_requested = True
    strategy = trainer_options.pop("strategy", None)
    if isinstance(strategy, dict):
        strategy = DeepSpeedStrategy(**strategy)
    checkpoint_options=dict(every_n_epochs=1,save_top_k=-1,save_last=True,
        save_weights_only=False,save_on_train_epoch_end=True)
    if config.get('checkpointing'):
        checkpoint_options.update(config['checkpointing'])
        if checkpoint_options.get('every_n_train_steps'):
            checkpoint_options.pop('every_n_epochs',None)
    callbacks = [
        LearningRateMonitor(logging_interval="step"),
        guard,
        ModelCheckpoint(
            dirpath=root / "resume",
            filename="epoch-{epoch:02d}-step-{step:07d}",
            **checkpoint_options,
        ),
    ]
    trainer = Trainer(
        **trainer_options,
        strategy=strategy or "auto",
        callbacks=callbacks,
        logger=CSVLogger(str(root), name="metrics"),
    )
    trainer.fit(runner, datamodule=data, ckpt_path=args.resume)


if __name__ == "__main__":
    main()
