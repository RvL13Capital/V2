"""AIv4 data_acquisition module."""

# Import modern sources for easier access
try:
    from . import sources
    from . import services
except ImportError:
    # Sources/services may not be available in all environments
    pass
