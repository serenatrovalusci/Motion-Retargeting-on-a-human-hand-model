import torch
import numpy as np
import joblib
from MR_FCNN import HandPoseFCNN

# ----------------------------
# 2. Load model + scaler
# ----------------------------
model = HandPoseFCNN()
model.load_state_dict(torch.load('best_model.pth'))
model.eval()

scaler_y = joblib.load("scaler_y.save")  # Make sure you saved it during training

# ----------------------------
# 3. Prediction function
# ----------------------------
def predict_joint_angles(closure_input):
    closure_input = np.round(closure_input, 3)  # round like in training
    x = torch.FloatTensor([closure_input])      # shape (1, 3)
    with torch.no_grad():
        pred = model(x).numpy()                 # shape (1, 27)
    angles = scaler_y.inverse_transform(pred)   # back to degrees
    return angles[0]                            # shape (27,)

# ----------------------------
# 4. Test it with sample closure
# ----------------------------
if __name__ == "__main__":
    # Example: thumb = 0.5, index = 0.8, middle = 0.3
    closure_input = [0.643, 0.533, 0.858]
    
    predicted_angles = predict_joint_angles(closure_input)

    print(" Input closure:", closure_input)
    print(" Predicted joint angles (degrees):")
    for i in range(9):
        base = i * 3
        print(f"Joint {i}: X={predicted_angles[base]:.2f}, Y={predicted_angles[base+1]:.2f}, Z={predicted_angles[base+2]:.2f}")
