import matplotlib.pyplot as plt

filename_original = 'log_RTX3060_kaparthy_bs_3'
filename_modified = 'log_RTX3060_optimalbatchsize128' 
title = 'RTX3060_compare'

window_size = 100

# Open the text file for reading
with open(filename_original + '.txt', 'r') as file:
    # Initialize empty lists to store data
    iterations_original = []
    losses_original = []
    # Iterate over each line in the file
    for line in file:
        # Split the line into words
        words = line.split()
        #print(words)
        # If the line contains the iteration number and loss value
        if words[2] == 'loss' and words[0] == 'iter':
            # Add the iteration number and loss value to their respective lists
            iterations_original.append(int(words[1].strip(':')))
            losses_original.append(float(words[3].strip(',')))

# the average loss value in the last 200 iterations of a training loop
last_200_losses = losses_original[-200:]
avg_loss = sum(last_200_losses) / len(last_200_losses)
print(f"Average kaparthy loss in last 200 iterations: {avg_loss}")

# do the same thing for original model
with open(filename_modified + '.txt', 'r') as file:
    # Initialize empty lists to store data
    iterations_modified = []
    losses_modified = []
    # Iterate over each line in the file
    for line in file:
        # Split the line into words
        words = line.split()
        #print(words)
        # If the line contains the iteration number and loss value
        if words[2] == 'loss' and words[0] == 'iter':
            # Add the iteration number and loss value to their respective lists
            iterations_modified.append(int(words[1].strip(':')))
            losses_modified.append(float(words[3].strip(',')))

# the average loss value in the last 200 iterations of a training loop
last_200_losses = losses_modified[-200:]
avg_loss = sum(last_200_losses) / len(last_200_losses)
print(f"Average crammed loss in last 200 iterations: {avg_loss}")


# calculate rolling mean and std
import pandas as pd
def rolling_mean_std(values, window_size):
    """
    Calculates the rolling mean and standard deviation of a list of values.
    :param values: A list of numerical values.
    :param window_size: The size of the window to use for the rolling calculation.
    :return: A tuple containing two lists: the rolling mean and standard deviation.
    """
    # Convert the values list to a pandas Series
    series = pd.Series(values)
    # Calculate the rolling mean using the specified window size
    rolling_mean = series.rolling(window=window_size).mean()
    # Calculate the rolling standard deviation using the specified window size
    rolling_std = series.rolling(window=window_size).std()
    # Return the rolling mean and standard deviation as a tuple of lists
    return (rolling_mean.tolist(), rolling_std.tolist())


rolling_mean_original, rolling_std_original = rolling_mean_std(losses_original, window_size)
#print("Rolling mean:", rolling_mean_original)
#print("Rolling std:", rolling_std_original)

rolling_mean_modified, rolling_std_modified = rolling_mean_std(losses_modified, window_size)
#print("Rolling mean:", rolling_mean_modified)
#print("Rolling std:", rolling_std_modified)



# Find the index of the lowest loss
# min_loss_idx = losses.index(min(losses))
# Add a red dot for the lowest loss
fig, ax = plt.subplots()
#ax.plot(min_loss_idx, losses[min_loss_idx], 'ro', label=f"min = ({min_loss_idx}, {losses[min_loss_idx]})")

# Create a line plot with iteration number on the x-axis and loss value on the y-axis
#ax.plot(rolling_mean_original, label='Kaparthy')
#ax.plot(rolling_mean_modified, label='Crammed')
ax.plot(rolling_std_original, label='Kaparthy')
ax.plot(rolling_std_modified, label='Crammed')

ax.set_yscale('log')
ax.grid(True)
ax.set_xlabel('Iteration Number ')
ax.set_ylabel('Loss rolling std')
fig.suptitle(title)
ax.legend()
fig.savefig('rolling_std.png', bbox_inches='tight')
