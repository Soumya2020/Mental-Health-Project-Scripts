import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset




class SimpleFCNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(SimpleFCNN, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, output_size)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        out = self.sigmoid(out)
        return out



def train_model(hidden_size, epochs, batch_size, learning_rate, patience, train_data, val_data):
    input_size = train_data[0].shape[1]
    output_size = 1  # Binary classification (0 or 1)

    # Initialize FCNN model, loss function, and optimizer
    model = SimpleFCNN(input_size, hidden_size, output_size)
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    train_loader = DataLoader(TensorDataset(*train_data), batch_size=batch_size, shuffle=True)# train dataloader
    val_loader = DataLoader(TensorDataset(*val_data), batch_size=batch_size, shuffle=False) # validation dataloader

    best_val_loss = float('inf')
    patience_counter = 0

    for epoch in range(epochs):
        model.train()
        epoch_loss = 0
        
        ## training process
        for X_batch, y_batch in train_loader:
            optimizer.zero_grad()              # Clears the gradients of all optimized parameters
            outputs = model(X_batch).flatten() # Forward pass through the model
            loss = criterion(outputs, y_batch)  #loss between predicted outputs and actual labels
            loss.backward()    # Backward pass to calculate gradients
            optimizer.step()   # update model parameters
            epoch_loss += loss.item()   #Accumulates the loss for each epoch 

        print(f'Epoch [{epoch+1}/{epochs}], Loss: {epoch_loss/len(train_loader):.4f}')

        model.eval()   # put model in evaluation model that disable batch normalization
        val_loss = 0
        # Loops through each batch in the val_loader without calculating gradients to compute average loss over validation batches
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                val_output = model(X_batch).flatten()
                val_loss += criterion(val_output, y_batch).item()
                
        val_loss /= len(val_loader)
        print(f'Validation Loss: {val_loss:.4f}')
        
        # check fr model improvement
        # If the current val_loss is lower than best_val_loss, update best_val_loss and reset patience_counter.
        # Save the model's parameters to best_fcnn_model.pth
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            # torch.save(model.state_dict(), 'best_fcnn_model.pth')
            
        # Patience Counter Increment:
        # If there's no improvement in validation loss, increment the patience_counter.
        # If the patience_counter reaches the patience limit, early stopping is triggered
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print("Early stopping triggered.")
                break
            
    # load best model and return it. Then prediction will be done using best model returned
    model.load_state_dict(torch.load('best_fcnn_model.pth'))
    return model


