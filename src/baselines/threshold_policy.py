class ThresholdPolicy:
    def __init__(self, intent_threshold=0.3, retrieval_threshold=0.3):
        self.intent_threshold = intent_threshold
        self.retrieval_threshold = retrieval_threshold
        
    def determine_action(self, intent, intent_confidence, retrieval_similarity):
        # Escalate on high-risk intents unconditionally
        if intent in ['account_security_activation', 'hardware_damage_repair', 'order_delivery_inquiry', 'other_unclear_context_dependent']:
            return 'ESCALATE', f"High-risk intent: {intent}"
            
        if intent_confidence < self.intent_threshold:
            return 'ESCALATE', f"Low intent confidence: {intent_confidence:.2f} < {self.intent_threshold}"
            
        if retrieval_similarity < self.retrieval_threshold:
            return 'ESCALATE', f"Low retrieval similarity: {retrieval_similarity:.2f} < {self.retrieval_threshold}"
            
        return 'AUTO_HANDLE', "Confidence and similarity above thresholds."
