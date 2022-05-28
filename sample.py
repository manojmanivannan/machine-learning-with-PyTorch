from prediction_models.generic import *


true_b = 1
true_z = 3
true_w = 2

# Number of samples
N = 200
# And make a convenient variable to remember the number of input columns
n_features = 2

# Data Generation
np.random.seed(42)
x = np.random.rand(N, n_features)

# y = true_b +( true_w * x ) + noise
y = true_b + (true_w * x[:,0]) +  (.1 * np.random.randn(N, )) + (true_z * x[:,1])
y = y.reshape(-1,1) # or y = y[...,None]




# Shuffles the indices
idx = np.arange(N)
np.random.shuffle(idx)

# Uses first 80 random indices for train
train_idx = idx[:int(N*.8)]
# Uses the remaining indices for validation
val_idx = idx[int(N*.8):]

# Generates train and validation sets
x_train, y_train = x[train_idx], y[train_idx]
x_val, y_val = x[val_idx], y[val_idx]

torch.manual_seed(13)

# Builds tensors from numpy arrays BEFORE split
x_tensor = torch.from_numpy(x).float()
y_tensor = torch.from_numpy(y).float()


# Builds dataset containing ALL data points
dataset = TensorDataset(x_tensor, y_tensor)


# Performs the split
ratio = .8
n_total = len(dataset)
n_train = int(n_total * ratio)
n_val = n_total - n_train

train_data, val_data = random_split(dataset, [n_train, n_val])

# Builds a loader of each set
train_loader = DataLoader(dataset=train_data, batch_size=16, shuffle=True)
val_loader = DataLoader(dataset=val_data, batch_size=16)


# Sets learning rate - this is "eta" ~ the "n" like Greek letter
lr = 0.1

torch.manual_seed(42)
# Now we can create a model and send it at once to the device
model = nn.Sequential(nn.Linear(n_features, 1))

# Defines a SGD optimizer to update the parameters
# (now retrieved directly from the model)
optimizer = optim.SGD(model.parameters(), lr=lr)

# Defines a MSE loss function
loss_fn = nn.MSELoss(reduction='mean')

print(model.state_dict())

sbs = StepByStep(model, loss_fn, optimizer)
sbs.set_loaders(train_loader,val_loader)
sbs.set_tensorboard('classy')

sbs.train(n_epochs=200)

print(model.state_dict())