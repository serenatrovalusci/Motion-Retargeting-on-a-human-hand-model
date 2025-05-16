from HandPoseClass import *
from train import *

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

def train_autoencoder(model, train_loader, test_loader, optimizer, scheduler, epochs=300, save_path="Autoencoder.pth"):
    train_losses, test_losses = [], []
    best_loss, best_model_state = float('inf'), None

    for epoch in range(epochs):
        model.train()
        train_loss = sum_step_loss_VAE(model, train_loader, optimizer=optimizer, training=True)

        model.eval()
        test_loss = sum_step_loss_VAE(model, test_loader, training=False)

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


closure_columns = ['ThumbClosure', 'IndexClosure', 'MiddleClosure', 'ThumbAbduction']
fix_indices = [0, 1, 2, 3, 4, 5, 6, 7, 8, 13, 14, 16, 17, 25, 26, 34, 43]
Y,scaler = load_data("hand_dataset_all_fingers.csv", closure_columns, fix_indices)

joblib.dump(scaler, "scaler_VAE.save")

input_dim = Y.shape[1]

dataset = TensorDataset(torch.FloatTensor(Y))  

print(f"Dataset size after outlier removal: {len(dataset)}")

dataloader = DataLoader(dataset, batch_size=64, shuffle=True)

train_loader, test_loader = train_test_split(Y, test_size=0.2, random_state=42)
train_loader = DataLoader(TensorDataset(torch.FloatTensor(train_loader)), batch_size=64, shuffle=True)
test_loader = DataLoader(TensorDataset(torch.FloatTensor(test_loader)), batch_size=64)

model = HandPoseVAE(input_dim=input_dim, latent_dim=20)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=10, factor=0.5, verbose=True)
#loss_fn = nn.MSELoss()

train_autoencoder(
    model=model,
    train_loader=train_loader,
    test_loader=test_loader,
    optimizer=optimizer,
    scheduler=scheduler,
    #loss_fn=loss_fn,
    epochs=300,
    save_path="HandPoseVVAE.pth"
)
