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

1. Start PLAY MODE
2. From Hierarchy and go to: WEART-> Hands -> WEARTLeftHand -> go to Inspector -> Add Component -> Script -> Hand Data Logger
3. From Hierarchy -> Hands-> WEARTLeftHand -> HandRig->HandRoot->DEF-hand.R/DEF-thumb.01.R/DEF-thumb.02.R/DEF-thumb.03.R select LeftHapticThumb and slide it to "Thumb Closure" in Lefthand Inspector in Hand Data Logger section (look for LeftHapticIndex and LeftHapticMiddle, respectively) 
4. From Hierarchy -> Hands-> WEARTLeftHand -> HandRig->HandRoot->DEF-hand.R/ select DEF-thumb.01.R and slide it to "Thumb 1" in the LeftHand Inspector in the Hand Data Logger section (look for all the remaining joints and do the same)
5. You should see from the Unity console that the CSV file has been created, you can start moving your hand, data is being registered 
6. Once you stop the PLAY MODE, the CSV file is saved
7. You can find the CSV file in AppData(W + R) -> LocalLow-> DefaultCompany-> Unity Project folder

# Neural Network 

# Hand Synergies

# Control Law


   

