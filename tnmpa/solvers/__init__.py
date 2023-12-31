import importlib.metadata

__version__ = importlib.metadata.version("tnmpa")

from . import quimb_vbp
from .message_passing_ksat import (
    BeliefPropagation,
    SurveyPropagation,
    TensorBeliefPropagation,
    TensorSurveyPropagation,
    TwoNormBeliefPropagation,
)
from .walksat import WalkSAT
