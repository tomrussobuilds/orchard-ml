"""
Dataset Domain Modules.

Organizes dataset metadata by domain (medical, space, etc.) to support
multi-domain framework usage while maintaining single source of truth pattern.
"""

from .medical import REGISTRY_28 as MEDICAL_28
from .medical import REGISTRY_224 as MEDICAL_224
from .space import REGISTRY_224 as SPACE_224

# Extension point: register a new domain here
# from .your_domain import REGISTRY_224 as YOUR_DOMAIN_224
# Then add YOUR_DOMAIN_224 to __all__ and merge it into the wrapper
# (see orchard/core/metadata/wrapper.py)

__all__ = [
    "MEDICAL_28",
    "MEDICAL_224",
    "SPACE_224",
]
