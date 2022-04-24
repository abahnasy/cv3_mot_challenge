from yacs.config import CfgNode

def get_cfg() -> CfgNode:
    """
    Get a copy of the default config.
    Returns:
        a YACS CfgNode instance.
    """
    from .defaults import _C

    return _C.clone()