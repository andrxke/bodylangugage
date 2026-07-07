"""
Pepper robot feedback module.

Provides post-session feedback delivery where Pepper replicates detected
body language gestures and verbally explains what it observed.

Components:
  - aggregator:  Accumulates gesture results into a FeedbackReport.
  - gestures:    Predefined Pepper pose presets for each gesture.
  - speech:      Text-to-speech driver (real + simulated).
  - motion:      Joint motion driver (real + simulated).
  - controller:  Orchestrates the full feedback delivery sequence.
"""

from feedback.aggregator import FeedbackAggregator, FeedbackReport
from feedback.controller import PepperFeedbackController

__all__ = [
    "FeedbackAggregator",
    "FeedbackReport",
    "PepperFeedbackController",
]
