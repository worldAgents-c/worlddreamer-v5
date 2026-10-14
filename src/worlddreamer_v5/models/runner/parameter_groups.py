# Copyright (C) 2026 World Agents Corporation.
NEW_PARAMETER_MARKERS = ("triplane", "field_render_head", "agent_field_injector", "spatial_action_gates")
NO_DECAY_MARKERS = (
    "bias",
    "norm",
    "ln",
    "rotary_emb",
    "adaln",
    "embedding",
    "query_base",
    "communication_gates",
)


def parameter_groups(parameters, base_lr=1.0, weight_decay=0.1):
    parameters = [(name, tensor) for name, tensor in parameters if tensor.requires_grad]
    groups = []
    for is_new, decay in ((False, True), (False, False), (True, True), (True, False)):
        selected = [
            tensor
            for name, tensor in parameters
            if any(marker in name for marker in NEW_PARAMETER_MARKERS) == is_new
            and (not any(marker in name.lower() for marker in NO_DECAY_MARKERS))
            == decay
        ]
        if selected:
            groups.append(
                {
                    "params": selected,
                    "weight_decay": weight_decay if decay else 0.0,
                    "lr": base_lr * (5.0 if is_new else 1.0),
                    "worlddreamer_v5_group": "spatial" if is_new else "backbone",
                }
            )
    return groups
