"""Local assistive-vision orchestration for the QCS8550 AIBOX.

This package deliberately treats the existing one-engine QNN perception service
as an external, immutable safety-critical dependency.  Speech and VLM services
are optional clients around that service; neither may alter its graph.
"""

__all__ = ["__version__"]
__version__ = "0.1.0"
