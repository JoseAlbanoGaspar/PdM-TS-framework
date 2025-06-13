from .optimusPipe import optimize_pipeline
from .transformers import (
    ImputationWrapper,
    FeatureExtractorWrapper,
    FeatureSelectionWrapper,
    RegularityResampler,
    NormalizationWrapper
)
from .metamodel import MetaClassifier

__version__ = '0.1.0'
__author__ = 'José Albano de Almeida Gaspar'

__all__ = [
    'optimize_pipeline',
    'ImputationWrapper',
    'FeatureExtractorWrapper',
    'FeatureSelectionWrapper',
    'RegularityResampler',
    'NormalizationWrapper',
    'MetaClassifier'
]