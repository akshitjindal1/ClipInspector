from .clip_inspector import ClipInspector


def create_defense(defense_type, model, dataset_details, config, device, output_dir=None):
    if defense_type != 'clip_inspector':
        raise ValueError(f"Unknown defense type: {defense_type}. Expected clip_inspector.")
    return ClipInspector(model, dataset_details, config, device, output_dir)

