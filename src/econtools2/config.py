"""Course-level settings. Each course calls `configure(...)` once at the top of a notebook."""
from dataclasses import dataclass, replace


@dataclass(frozen=True)
class Config:
    # URL prefix for graph images referenced in the HTML (online_*) versions of problems,
    # e.g. "/CSUN-Econ-310/assets/images/graphs" for a Jekyll course site.
    image_url_prefix: str = "."


_config = Config()


def get_config() -> Config:
    return _config


def configure(**kwargs) -> Config:
    global _config
    _config = replace(_config, **kwargs)
    return _config
