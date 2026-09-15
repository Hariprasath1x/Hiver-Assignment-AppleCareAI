class MajorityClassifier:
    def __init__(self, dev_df):
        # Find majority intent in dev data
        self.majority_intent = dev_df['expected_intent'].value_counts().index[0]
        self.generic_reply = "Thanks for contacting Apple Support. Please provide more details so we can assist you."
        
    def predict(self, text):
        return {
            'intent': self.majority_intent,
            'action': 'ESCALATE',
            'reply': self.generic_reply,
            'confidence': 1.0,
            'reason': 'Majority Baseline policy.'
        }
