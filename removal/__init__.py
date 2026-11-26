from .fixed_trigger import FixedTriggerRemover


def create_remover(remover_type, *args, **kwargs):
    removers = {
        "fixed_trigger": FixedTriggerRemover,
    }
    if remover_type not in removers:
        raise ValueError(
            f"Unknown remover type: {remover_type}. Available types: {list(removers.keys())}"
        )
    return removers[remover_type](*args, **kwargs)
