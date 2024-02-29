from deepeval.metrics import AnswerRelevancyMetric, BiasMetric, FaithfulnessMetric

# Configure metrics here through their parameters
metrics = [
    AnswerRelevancyMetric(),
    BiasMetric(),
    FaithfulnessMetric()
]
