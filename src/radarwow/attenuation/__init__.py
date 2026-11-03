"""radar-wow stage 4: attenuation and differential attenuation correction (rain)."""

from .methods import Bringi2001SelfConsistent, LinearPhiAttenuation, Testud2000ZPHI, zphi
from .testing import synthetic_attenuated_sweep

__all__ = ["Bringi2001SelfConsistent", "LinearPhiAttenuation", "Testud2000ZPHI", "synthetic_attenuated_sweep", "zphi"]
