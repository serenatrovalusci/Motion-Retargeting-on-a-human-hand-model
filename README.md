# Hand Motion Reconstruction from Minimal Input through Latent-Space Learning

We address the problem of real-time motion retargeting for a virtual human hand using the WeART TouchDIVER G1 haptic glove and a data-driven approach based on a neural network. In order to exploit the Hand Synergies and improve both generalization and efficiency, we apply dimensionality reduction techniques such as Principal Component Analysis (PCA) and Autoencoders. 

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

# Steps for REAL-TIME PREDICTION using NN (FCNN or Transformer)
   a. Direct PCA Output:  We reduced the dataset using PCA and trained the NN to predict the PCA components. The joint predictions are obtained by using the PCA inverse transform.
   
      1. In the folder "training_results/training_synergies_results" you can find the weights and parameters for the trained models (FCNN or Transformer) using different number of PCA components: 10,                 15, 30, 45.
      
      2. To replicate the results, you can run "python main_synergies.py --info_path training_results\training_synergies_results\training_20250521_150116\training_info.txt" (example).
      
      3. Once you obtain the message: "Server ready (Model: {model_type} | Fixed indices: {fix_indices})...", you can start PLAY MODE on Unity and run the hand simulation.

   b. PCA-Based Loss Only:  We used the whole dataset to train the NN, and we used the dimensionality reduction only in the loss function (PCA). The NN outputs the whole set of joints predictions.
   
      1. In the folder "training_results/training_losspca_results" you can find the weights and parameters for the trained models (FCNN or Transformer) using different number of PCA components: 10,                   15, 30, 45.
      
      2. To replicate the results, you can run "python main.py --info_path training_results/training_losspca_results/training_20250606_191433/training_info.txt" (example). 
      
      3. Once you obtain the message: "Server ready (Model: {model_type} | Fixed indices: {fix_indices})...", you can start PLAY MODE on Unity and run the hand simulation.

   c. Autoencoder-Based Loss:  We used the whole dataset to train the NN, and we used the dimensionality reduction only in the loss function (Autoencoder). Also added a constraint term in the loss to              prevent unrealistic joint configurations. The NN outputs the whole set of joints predictions.
   
      1. In the folder "training_results/training_latentspace_results" you can find the weights and parameters for the trained models (FCNN or Transformer) using different dimensions for the latent space:            10, 15, 30, 45.
      
      2. To replicate the results, you can run "python main.py --info_path training_results/training_latentspace_results/FCNN/training_20250606_181904/training_info.txt" (example).
      
      3. Once you obtain the message: "Server ready (Model: {model_type} | Fixed indices: {fix_indices})...", you can start PLAY MODE on Unity and run the hand simulation.
   
 # Training files

   1. Direct PCA Output --> train_synergies.py
   2. PCA-Based Loss Only --> train_losspca.py
   3. Autoencoder-Based Loss --> train_latentspace_input.py
   4. Autoencoder (alone) --> train_AE_VAE.py

 



   

