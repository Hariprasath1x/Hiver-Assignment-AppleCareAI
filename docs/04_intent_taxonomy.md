# Phase 4: Intent Discovery & Taxonomy

## 1. Discovery Methodology
The intent taxonomy was designed by synthesizing the data-driven KMeans topical clusters from Phase 3 with the operational requirement of "How would an AppleSupport agent actually handle this?". 
We explicitly avoided importing generalized frameworks (like Banking77) and avoided overfitting to temporary product phenomena (like the iOS 11 'i' bug).

## 2. Candidate Clusters vs Operational Intents
In Phase 3, we identified clusters like:
- **Cluster 1**: The iOS 11 'i' autocorrect bug
- **Cluster 2**: General iOS update lag/crashing
- **Cluster 3**: Apple Music / iTunes

While the 'i' bug was massively prevalent historically, creating an `ios11_i_bug` intent violates operational validity because it is a temporary software bug. Instead, it was generalized into **Keyboard & Text Input Issue**, which allows the agent to handle any future text rendering, predictive text, or autocorrect bugs with a standard text-replacement or software update workflow.

## 3. Final Taxonomy
The final taxonomy consists of exactly **10 Operational Intents**:
1. `battery_power_issue`
2. `software_system_issue`
3. `keyboard_text_input_issue`
4. `app_store_media_services`
5. `hardware_damage_repair`
6. `account_security_activation`
7. `network_connectivity`
8. `feature_inquiry_how_to`
9. `order_delivery_inquiry`
10. `other_unclear_context_dependent`

## 4. Definitions, Examples, and Confusion Pairs

### 1. Battery & Power Issue
- **Definition**: Issues related to battery drain, poor battery life, device getting too hot, or failing to charge.
- **Example**: *"Tired of charging my phone for mad long for it to die in 30 minutes"*
- **Historical Support Behavior**: Asks to check Battery Health in Settings, points to a KB article on maximizing battery life.
- **Confusion Pair**: May be confused with *Software/System Issue* if a user says "the update ruined my battery".

### 2. Software & System Issue
- **Definition**: General software bugs, lag, freezing, apps crashing, or errors encountered during iOS/macOS updates.
- **Example**: *"This IOS 11.0.2 update is trash. Phone has been freezing all day"*
- **Historical Support Behavior**: Asks for current iOS version, suggests force restart, or asks if it happens in specific apps.
- **Confusion Pair**: Confused with *Hardware/Damage Repair* when a user says "my phone is broken" without specifying if it's physical or software.

### 3. Keyboard & Text Input Issue
- **Definition**: Issues specifically regarding typing, keyboard rendering, autocorrect, or predictive text.
- **Example**: *"can y’all fix this i issue please... its been 3 days now. And im tired of my i’s being typed as A’s"*
- **Historical Support Behavior**: Provides a temporary text-replacement workaround via Settings > General > Keyboard, or tells them to update to iOS 11.1.1.
- **Why Separate**: Operationally, typing issues (especially severe ones) have their own immediate workarounds (Text Replacement) that differ completely from standard app crash troubleshooting.

### 4. App Store & Media Services
- **Definition**: Issues relating to downloading apps, Apple Music, Podcasts, iTunes credit, or iCloud storage.
- **Example**: *"I can’t update my apps so some of them I can’t go on anymore"*
- **Historical Support Behavior**: Often deflects to a specialized iTunes/Media support team via DM.
- **Why Separate**: Media and purchasing issues fall under a different business unit (Services) than hardware/software troubleshooting.

### 5. Hardware, Damage & Repair
- **Definition**: Physical hardware failures, broken screens, water damage, or unresponsive hardware components.
- **Example**: *"Updates to iOS 11.3 and now my Touch ID isn’t working."*
- **Historical Support Behavior**: Requires inspecting the device. Points customer to locate an Apple Store or Authorized Service Provider.
- **Why Separate**: Cannot be fixed via software troubleshooting. Requires physical repair booking.

### 6. Account, Security & Activation
- **Definition**: Issues logging into Apple ID, 2-factor authentication, activation lock, forgotten passwords, or unauthorized charges.
- **Example**: *"can’t activate my iPhone due to server issues heellllllp"*
- **Historical Support Behavior**: Immediate escalation to DM or secure link due to privacy rules.
- **Why Separate**: High-risk category. The AI agent should almost always escalate these to avoid leaking PII.

### 7. Network & Connectivity
- **Definition**: Issues connecting to Wi-Fi, cellular data, Bluetooth devices, or AirDrop.
- **Example**: *"I can’t connect to my WiFi even when I’m close to the router."*
- **Historical Support Behavior**: Provides KB article on "Reset Network Settings" or checking carrier updates.

### 8. Feature Inquiry & How-To
- **Definition**: Questions about how to use a feature, checking compatibility, or general non-broken inquiries.
- **Example**: *"how do I add my battery percentage to always show on iPhone X?"*
- **Historical Support Behavior**: Links directly to an Apple Support KB article explaining the feature.

### 9. Order & Delivery Inquiry
- **Definition**: Questions regarding purchasing, shipping status, Apple Store appointments, or returns.
- **Example**: *"My iPhone X order is “In Progress” Ppl who ordered after me are receiving their deliveries already !"*
- **Historical Support Behavior**: Requests order number in DM.

### 10. Other, Unclear & Context-Dependent
- **Definition**: Messages that lack sufficient context to categorize, follow-ups to previous conversations, or rants without an actionable problem.
- **Example**: *"Yes you did thx :)", "Just like that!!!", "DM?"*
- **Historical Support Behavior**: Asks clarifying questions ("Which OS are you on?"), deflects to DM, or acknowledges the thanks.
- **Why Separate**: Crucial for safety. The agent must recognize when it DOES NOT know what the customer is talking about, and escalate rather than guessing.

## 7. Taxonomy Trade-offs
- **Trade-off**: Grouping the 'i' bug under `keyboard_text_input_issue`. 
- **Reason**: The 'i' bug dominates the dataset. Creating an `ios11_i_bug` intent would yield 99% accuracy on this dataset but would fail in production a year later. The generic keyboard intent is future-proof.
- **Trade-off**: Creating the `other_unclear` bucket.
- **Reason**: We evaluated about 5% of our interactions as context-dependent or too short. A classifier forced to pick a valid technical issue for "Yes I did" will hallucinate.

## 8. Golden-Set Implications (Updated)
We revise our Phase 3 recommendation. The Golden Set must test the agent's ability to safely handle edge cases, not just its accuracy on standard queries.
We recommend a 200-example Golden Set containing:
- **100 "Normal" interpretable cases** (stratified across the 9 technical intents).
- **30 Context-dependent / Follow-up cases** (to test if the classifier correctly assigns `other_unclear`).
- **20 Escalation-sensitive cases** (specifically `account_security` and `hardware_damage`) to test the escalation policy.
- **20 URL/Image-only cases** (to verify the agent doesn't hallucinate context).
- **30 Rare intents** (like `order_delivery`).

The Golden Set will be manually constructed and completely isolated from the retrieval corpus.
