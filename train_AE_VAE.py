from HandPoseClass import *
from train_losspca import *
from datetime import datetime

def create_training_directory(base_dir="training_results/training_reduction_models"):
    # Create a unique directory name with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    training_dir = os.path.join(base_dir, f"training_{timestamp}")
    os.makedirs(training_dir, exist_ok=True)
    return training_dir

def save_training_info(training_dir, info_dict):
    # Save training info to a text file
    info_file = os.path.join(training_dir, "training_info.txt")
    with open(info_file, 'w') as f:
        for key, value in info_dict.items():
            f.write(f"{key}: {value}\n")

def load_data(csv_path, closure_columns, fix_indices, z_thresh=2.5):
    print("Loading Dataset from csv file...")
    data = pd.read_csv(csv_path)
    data.columns = data.columns.str.strip()
    data = data.iloc[:, 1:] # remuve the first column (time)
    joint_columns = [c for c in data.columns if c not in closure_columns]
    X = data[closure_columns].values
    Y = data[joint_columns].values

    print("Dataset size:", len(TensorDataset(torch.FloatTensor(Y))))

    _, Y = remove_outliers_zscore(X, Y, z_thresh)
    Y_sincos = generate_sincos_dataset(Y, fix_indices)

    scaler = StandardScaler().fit(Y_sincos)
    Y_scaled = scaler.transform(Y_sincos)

    print("All components:", Y_scaled.shape[1])

    return Y_scaled, scaler

def sum_step_loss_AE(model, loader, loss_fn, optimizer=None, training=False):
    total_loss = 0.0
    for batch in loader:
        x = batch[0]
        x_hat, _ = model(x)
        loss = loss_fn(x_hat, x) 

        if training:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        total_loss += loss.item() * x.size(0)
    return total_loss / len(loader.dataset)

def sum_step_loss_VAE(model, loader, optimizer=None, training=False):
    total_loss = 0.0
    for batch in loader:
        x = batch[0]
        #x_hat, _ = model(x) 
        recon_x, mu, logvar = model(x)
        loss = vae_loss(recon_x, x, mu, logvar)
        #loss = loss_fn(x_hat, x)

        if training:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        total_loss += loss.item() * x.size(0)
    return total_loss / len(loader.dataset)

def vae_loss(recon_x, x, mu, logvar):
    BCE = torch.nn.functional.mse_loss(recon_x, x, reduction='sum')
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return BCE + KLD

def train_autoencoder(model, train_loader, test_loader, optimizer, scheduler, loss_fn=None, epochs=300, save_path="Autoencoder.pth"):
    train_losses, test_losses = [], []
    best_loss, best_model_state = float('inf'), None

    # Detect model type based on forward signature
    is_vae = isinstance(model, HandPoseVAE)

    for epoch in range(epochs):
        model.train()
        if is_vae:
            train_loss = sum_step_loss_VAE(model, train_loader, optimizer=optimizer, training=True)
        else:
            train_loss = sum_step_loss_AE(model, train_loader, loss_fn, optimizer=optimizer, training=True)

        model.eval()
        if is_vae:
            test_loss = sum_step_loss_VAE(model, test_loader, training=False)
        else:
            test_loss = sum_step_loss_AE(model, test_loader, loss_fn, training=False)

        scheduler.step(test_loss)
        train_losses.append(train_loss)
        test_losses.append(test_loss)

        if test_loss < best_loss:
            best_loss = test_loss
            best_model_state = model.state_dict()
            torch.save(best_model_state, save_path)

        if epoch % 20 == 0:
            lr = optimizer.param_groups[0]['lr']
            print(f"Epoch {epoch:03d} | LR: {lr:.5f} | Train Loss: {train_loss:.6f} | Test Loss: {test_loss:.6f}")

    if best_model_state:
        model.load_state_dict(best_model_state)

    return train_losses, test_losses



def main():
    #### prompt python train_AE_VAE.py --model VAE --latent_dim 40 --epochs 300 --save_model my_ae_model.pth ####
    parser = argparse.ArgumentParser(description="Train AE or VAE for hand pose encoding.")
    parser.add_argument('--model', type=str, default='AE', choices=['AE', 'VAE'], help='Model type to train')
    parser.add_argument('--latent_dim', type=int, default=30, help='Latent space dimension')
    parser.add_argument('--epochs', type=int, default=300, help='Training epochs')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size')
    parser.add_argument('--save_model', type=str, default=None, help='Optional custom model save path')
    args = parser.parse_args()

    # Setup
    closure_columns = ['ThumbClosure', 'IndexClosure', 'MiddleClosure', 'ThumbAbduction']
    fix_indices = [0, 1, 2, 3, 4, 5, 6, 7, 8, 13, 14, 16, 17, 25, 26, 34, 43]
    training_start_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    training_dir = create_training_directory()

    # Data
    Y, scaler = load_data("dataset/hand_dataset_all_fingers.csv", closure_columns, fix_indices)
    joblib.dump(scaler, os.path.join(training_dir, "scaler_AE.save"))
    input_dim = Y.shape[1]

    Y_train, Y_test = train_test_split(Y, test_size=0.2, random_state=42)
    train_loader = DataLoader(TensorDataset(torch.FloatTensor(Y_train)), batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(TensorDataset(torch.FloatTensor(Y_test)), batch_size=args.batch_size)

    # Model selection
    if args.model == 'AE':
        model = HandPoseAE(input_dim=input_dim, latent_dim=args.latent_dim)
    else:
        model = HandPoseVAE(input_dim=input_dim, latent_dim=args.latent_dim)

    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=10, factor=0.5, verbose=True)
    loss_fn = nn.MSELoss()

    save_path = args.save_model or os.path.join(training_dir, f"HandPose{args.model}_{args.latent_dim}latent.pth")

    # Train
    train_losses, test_losses = train_autoencoder(
        model, train_loader, test_loader,
        optimizer, scheduler, loss_fn,
        epochs=args.epochs, save_path=save_path
    )

    # Info logging
    training_info = {
        "Model": args.model,
        "Latent Dimension": args.latent_dim,
        "Epochs": args.epochs,
        "Batch Size": args.batch_size,
        "Training Start Time": training_start_time,
        "Training End Time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "Dataset": "dataset/hand_dataset_all_fingers.csv",
        "Input Dimension": input_dim,
        "Best Test Loss": min(test_losses),
        "Final Learning Rate": optimizer.param_groups[0]['lr'],
        "Model Save Path": save_path
    }

    save_training_info(training_dir, training_info)

    print(f"Training complete. Model saved at: {save_path}")
    print("Training Info:")
    for k, v in training_info.items():
        print(f"{k}: {v}")

if __name__ == "__main__":
    main()



# closure_columns = ['ThumbClosure', 'IndexClosure', 'MiddleClosure', 'ThumbAbduction']
# fix_indices = [0, 1, 2, 3, 4, 5, 6, 7, 8, 13, 14, 16, 17, 25, 26, 34, 43]
# Y,scaler = load_data("dataset/hand_dataset_all_fingers.csv", closure_columns, fix_indices)

# joblib.dump(scaler, "scaler_AE.save")

# input_dim = Y.shape[1]

# dataset = TensorDataset(torch.FloatTensor(Y))  

# print(f"Dataset size after outlier removal: {len(dataset)}")

# dataloader = DataLoader(dataset, batch_size=64, shuffle=True)

# train_loader, test_loader = train_test_split(Y, test_size=0.2, random_state=42)
# train_loader = DataLoader(TensorDataset(torch.FloatTensor(train_loader)), batch_size=64, shuffle=True)
# test_loader = DataLoader(TensorDataset(torch.FloatTensor(test_loader)), batch_size=64)

# model = HandPoseAE(input_dim=input_dim, latent_dim=30)
# optimizer = torch.optim.Adam(model.parameters(), lr= 1e-3)
# scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=10, factor=0.5, verbose=True)
# loss_fn = nn.MSELoss()

# train_autoencoder(
#     model=model,
#     train_loader=train_loader,
#     test_loader=test_loader,
#     optimizer=optimizer,
#     scheduler=scheduler,
#     loss_fn=loss_fn,
#     epochs=500,
#     save_path="HandPoseAE_2.pth"
# )