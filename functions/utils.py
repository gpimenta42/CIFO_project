import math 

def config_to_key(config):
    """Turn a parameter config dictionary into a tuple key for fast comparison."""
    def safe(val):
        return None if isinstance(val, float) and math.isnan(val) else val

    return (
        safe(config["pop_size"]),
        safe(config["mutation"]),
        safe(config["crossover"]),
        safe(config["selection"]),
        safe(config["selection_param"]),
        safe(config["elitism"]),
        safe(config["max_gen"]),
        round(safe(config["xo_prob"]), 4),
        round(safe(config["mut_prob"]), 4),
    )
