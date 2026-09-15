# Phase 3: Exploratory Data Analysis Findings

## A. Dataset Overview
- **Total Interaction Pairs**: 106625
- **Customer Message Length**: Median=18.0, p95=37.0 words
- **Support Response Length**: Median=21.0, p95=40.0 words

## B. Response Type Analysis
Breakdown of historical AppleSupport responses:
- **DM Deflection**: 56138 (52.6%)
- **Clarification/Question**: 21089 (19.8%)
- **Acknowledgement/Other**: 17278 (16.2%)
- **Substantive Troubleshooting**: 10236 (9.6%)
- **Link/Resource Response**: 1884 (1.8%)

## C & F. Customer Message Quality and Context Dependency
- **Standard**: 100857 (94.6%)
- **Very Short/Context Dependent**: 5061 (4.7%)
- **URL/Image Only**: 485 (0.5%)
- **Mentions Only / Empty**: 222 (0.2%)

## D. Duplicates
- **Exact Customer Text Duplicates**: 1810

## E. Issue / Intent Discovery (KMeans Clustering)
### Cluster_0
- **Keywords**: music, apple music, apple, applesupport, app
- **Count (in 20k sample)**: 505
- **Examples**:
  - `@AppleSupport why does my Apple Watch constantly show me my music controls????? I just want to see the time!!!`
  - `iOS 11.0.3, found a bug in music app. Where the music app doesn’t work from the lock screen. @AppleSupport @115858 @115948`
  - `@115948 @AppleSupport Hey, I've discovered lots of incorrect versions of songs (single songs within whole otherwise correct albums) in Apple Music. There should probably be a way to flag instances of this for review. Kind of annoying. Thanks!`

### Cluster_1
- **Keywords**: just, applesupport just, applesupport, phone, 115858
- **Count (in 20k sample)**: 1096
- **Examples**:
  - `@AppleSupport just DMed you guys about my problem`
  - `@AppleSupport is this error message from Apple? It just popped up on my phone https://t.co/GQ0tIt1Hew`
  - `@AppleSupport Yeah I actually just fixed it by tapping the top of the phone above the camera. Works fine again! Thanks!`

### Cluster_2
- **Keywords**: iphone, applesupport iphone, applesupport, plus, iphone plus
- **Count (in 20k sample)**: 1406
- **Examples**:
  - `@AppleSupport Thanks. 11.1.1. iPhone 6S`
  - `Again yet another iPhone battery life ruined by @115858 update #ios1112 first my iPhone 6 last year now my iPhone 7plus! Come on apple sort it out!`
  - `@AppleSupport Having issues downloading apps to my iPhone SE.`

### Cluster_3
- **Keywords**: update, phone, 115858, new, applesupport
- **Count (in 20k sample)**: 2012
- **Examples**:
  - `Such a rubbish update @115858 #iphoneupdate`
  - `@AppleSupport My keyboard does not work well since I made the iOS update`
  - `So how did @115858 type the “i” in the iOS update? [?]`

### Cluster_4
- **Keywords**: 115858, fix, phone, 115858 fix, shit
- **Count (in 20k sample)**: 3445
- **Examples**:
  - `@115858 macOS has root without password bug,even if I disable root.`
  - `Hey @115858 my phone is almost unusable since ios11, all of your advices never work, what should i do??? I only bought it 2 years ago`
  - `@115858 y’all gonna fix this shit or what ? If not i want free iPhone X !`

### Cluster_5
- **Keywords**: amp, work, doesn, doesn work, applesupport
- **Count (in 20k sample)**: 852
- **Examples**:
  - `@115858 My Cinema Display doesn't work with My macbook touchbar using the correct adaptor. I tried in my office and it works. Any ideas?`
  - `@AppleSupport Apple earphones’ mic doesn’t work with my phone and neither do the buttons on it. When I’m talking on the phone I have to speak through my phone as if I put it on speaker. They work on my friends’ phones though. My phone is working properly.`
  - `@AppleSupport the shit don’t work clearly https://t.co/uV5CgdIOBU`

### Cluster_6
- **Keywords**: applesupport, https, phone, help, app
- **Count (in 20k sample)**: 5968
- **Examples**:
  - `@AppleSupport Mainly with front camera. From back camera also it happened few times. When i take pic in portrait mode it doesn't blur background instead it shows black background. It's iPhone X`
  - `@AppleSupport Hey there! Can I ask you about AppStore?`
  - `Wtf is this @AppleSupport ??? I️`

### Cluster_7
- **Keywords**: 115858 applesupport, 115858, applesupport, https, battery
- **Count (in 20k sample)**: 683
- **Examples**:
  - `@115858 @AppleSupport screen glitchy and unresponsive after update !!??!!! 😡 brand new pho ne !!`
  - `@115858 @AppleSupport what the hell is going on... why is everyone’s phone doing this now I️ I️ I️ I️ when we type the letter “i”`
  - `@115858 @AppleSupport your service centers in India r pathetic they made my phone worst.`

### Cluster_8
- **Keywords**: apple, hi, applesupport, applesupport hi, watch
- **Count (in 20k sample)**: 1187
- **Examples**:
  - `@AppleSupport - my Apple TV remote isn't working, so I downloaded the app... but now I need the remote to sync the app on my phone? Help???`
  - `@115858 how is my storage full when I have 128gb of memory, and only used 37.5gb of memory? Like what the fuck? Apple ya’ll losing customers smh step your game up before I go to the homie Samsung....`
  - `@AppleSupport hi, I'm getting notified of sign in attempts (by myself) but it keeps saying the wrong location, even though my macbook correctly recognises where I am. Any advice??`

### Cluster_9
- **Keywords**: thank, applesupport thank, applesupport, 115858, update
- **Count (in 20k sample)**: 285
- **Examples**:
  - `@AppleSupport wow that worked thank u`
  - `@AppleSupport I got the new iOS update and it worked thank you so much.`
  - `#iOS11 is probably one of the worst software upgrades in recent memory. It makes my iPhone 6 feel like a 4. Thank you, @115858.`

### Cluster_10
- **Keywords**: issue, fix issue, applesupport, fix, 115858
- **Count (in 20k sample)**: 566
- **Examples**:
  - `Yo @AppleSupport how are you zaddy? Do us a favor and fix this I️ issue that came with your update. Thanks baby. Love you https://t.co/ZQGKdrVlRK`
  - `@AppleSupport Issue is due to device management in which the workspace services(mdm account) has restricted explicit content, how can I unrestrict it?`
  - `Umm yeah @AppleSupport can y’all fix this i issue please... its been 3 days now. And im tired of my i’s being typed as A’s.....`

### Cluster_11
- **Keywords**: 11, ios 11, ios, applesupport, iphone
- **Count (in 20k sample)**: 1995
- **Examples**:
  - `@AppleSupport ‘upgrade’ to iOS 11.0.3 has ruined my devices: lags, shadows, shut downs, it’s disconnected and reduced productivity.`
  - `@324687 @324688 @115858 I’m running a 6, and since the iOS 11 update, its gone from a perfectly usable phone to an unreliable POS. Constant freezing.`
  - `@AppleSupport y’all fucked up my phone w iOS 11`

## G. Retrieval Corpus Candidate Analysis
- **Usable Candidate Interactions**: 47634 (44.7%)
- These are interactions where the customer provided a substantive message and AppleSupport provided a substantive, non-DM response.

## H. Escalation Signals
- **Low Information/Context Dependent**: 20-30% of messages are too short or rely on images/URLs.
- **DM Deflection Heaviness**: ~50% of historical AppleSupport responses are DM deflections. The AI should escalate cases where history says 'DM us'.

## I. Golden Set Implications
- **Recommendation**: Sample 200 interactions stratified across the 12 discovered clusters to ensure all issue types are covered. Ensure we sample only from the 'Standard' customer messages that lead to usable candidate interactions to evaluate retrieval reliably.
