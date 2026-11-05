__all__ = ["core", "config"]


def __getattr__(name: str):
    if name in __all__:
        module = __import__(f"utils.{name}", fromlist=[name])
        globals()[name] = module
        return module
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
