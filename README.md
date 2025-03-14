# Motion-Retargeting-on-a-human-hand-model

This project explores the use of hand synergies to control a simulated hand in Unity more efficiently.

# Steps for Motion Retargeting with Built-In functions of Weart SDK: 

1. Download Unity Hub 3.0: https://unity.com/download
2. Download the Weart Unity SDK: https://weart.it/repository/downloads/unity/WEART_SDK_Unity_v2.0.0.zip and extract folder
3. Create New Project
4. Go to: Window->Package Manager-> + -> Add package from disk -> select extracted folder from downloaded zip
5. Go to: WEART -> Add Weart Startup Components
6. From Hierarchy and go to: WEART-> Hands -> WEARTLeftHand -> go to Inspector and set the Tracking Source as WeArtController
7. Start PLAY MODE 

# Steps for DATASET creation

1. From Hierarchy and go to: WEART-> Hands -> WEARTLeftHand -> go to Inspector -> Add Component -> Script -> Hand Data Logger
2. Select the joints fron WEARTLeftHand and slide them into the "Hand Data Logger (Script)" section
   

