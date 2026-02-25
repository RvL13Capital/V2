"""AIv4 training module."""

# Import modern training components
try:
    from . import models
    from . import services
except ImportError:
    # Models/services may not be available in all environments
    pass
