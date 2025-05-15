from train import *
from HandPoseClass import HandPoseAE

model = HandPoseAE(input_dim = 45, latent_dim= 20)
model.load_state_dict(torch.load("HandPoseAE.pth"))
model.eval()

scaler = joblib.load("scaler_AE.save")

def load_data_AE(csv_path, closure_columns, z_thresh= 2.5):
    print("Loading Dataset from csv file...")
    data = pd.read_csv(csv_path)
    data.columns = data.columns.str.strip()
    data = data.iloc[:, 1:] # remuve the first column (time)
    joint_columns = [c for c in data.columns if c not in closure_columns]
    X = data[closure_columns].values
    Y = data[joint_columns].values

    print("Dataset size:", len(TensorDataset(torch.FloatTensor(Y))))

    _, Y = remove_outliers_zscore(X, Y, z_thresh)

    scaler = StandardScaler().fit(Y)
    Y_scaled = scaler.transform(Y)

    print("All components:", Y_scaled.shape[1])

    return Y_scaled, scaler


closure_columns = ['ThumbClosure', 'IndexClosure', 'MiddleClosure', 'ThumbAbduction']
Y,_ = load_data_AE("hand_dataset_all_fingers.csv", closure_columns)


Y_prova, z_prova = model(torch.FloatTensor(Y[0]).unsqueeze(0))

print(Y[0])

print(Y_prova)
print(z_prova)