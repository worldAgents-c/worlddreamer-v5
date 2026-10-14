# Copyright (C) 2026 World Agents Corporation.
def initialization_audit(metadata, expected_contract, policy=None):
    source = metadata['robot_action_contract']
    keys = ('state_dim', 'action_dim', 'state_history_length', 'action_length',
            'action_target', 'normalization_dataset', 'action_layout', 'state_layout')
    for key in keys:
        if source.get(key) != expected_contract.get(key):
            raise ValueError(f'initialization semantics mismatch: {key}')
    return {'source_model_sha256': metadata['model_sha256'],
            'optimizer_state_restored': False}
