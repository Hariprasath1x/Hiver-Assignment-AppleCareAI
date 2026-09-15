# Golden Set Annotation Guidelines

These guidelines define how the lead engineer manually annotated the 200-example Golden Evaluation Set.

## Core Principle
The golden set establishes the ground truth for **what the AI agent SHOULD do**, not merely what AppleSupport did historically. While historical actions inform the annotation, the primary focus is safe, evidence-based customer support.

## Task 1: Intent Classification
Assign exactly one of the 10 operational intents. 
If an example genuinely cannot be classified from the available text (e.g. "Yes I did"), assign `other_unclear_context_dependent`. Do NOT guess or force an intent.

### Borderline Cases
- **Software Bug vs Keyboard Bug**: If the bug *specifically* relates to typing, autocorrect, or the iOS 11 'i' issue, use `keyboard_text_input_issue`. If it is a general crash or lag, use `software_system_issue`.
- **Hardware vs Software**: If the customer says "my phone is broken", classify as `software_system_issue` unless physical damage (screen, water) or hardware buttons/Touch ID are explicitly mentioned (`hardware_damage_repair`).
- **Account Lock vs App Store**: If a user cannot download an app because their *account is disabled for security reasons*, classify as `account_security_activation`. If it's a generic App Store error, use `app_store_media_services`.

## Task 2: Action Assignment
Assign either `AUTO_HANDLE` or `ESCALATE`.

### AUTO_HANDLE
Assign this if the intent is a standard technical issue (`battery_power_issue`, `software_system_issue`, `network_connectivity`, `feature_inquiry_how_to`, `keyboard_text_input_issue`) AND there is sufficient context to attempt a retrieval-augmented generation response.

### ESCALATE
Assign this if ANY of the following are true:
1. **Security/Privacy Risk**: (`account_security_activation`) - Involves Apple ID, passwords, or unauthorized charges.
2. **Physical Inspection Required**: (`hardware_damage_repair`) - Involves cracked screens or broken hardware.
3. **Backend Access Required**: (`order_delivery_inquiry`) - Involves tracking packages or Apple Store appointments.
4. **Insufficient Context**: (`other_unclear_context_dependent`) - Message is < 4 words, a follow-up ("yes"), or just an image/URL. 

## Conflict Resolution
If an interaction overlaps (e.g., "My screen is cracked and I forgot my password"), prioritize the highest-risk escalation intent (in this case, `account_security_activation`).
