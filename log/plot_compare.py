import matplotlib.pyplot as plt

filename_original = 'log_RTX3060_kaparthy_bs_3'
filename_modified = 'log_RTX3060_optimalbatchsize128' 
title = 'RTX3060_compare'

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


## plot 
#TODO
#have loss for original model as well


# Find the index of the lowest loss
#min_loss_idx = losses.index(min(losses))
# Add a red dot for the lowest loss
fig, ax = plt.subplots()
#ax.plot(min_loss_idx, losses[min_loss_idx], 'ro', label=f"min = ({min_loss_idx}, {losses[min_loss_idx]})")

# Create a line plot with iteration number on the x-axis and loss value on the y-axis
ax.plot(iterations_original, losses_original, label='kaparthy')
ax.plot(iterations_modified, losses_modified, label='Crammed')

ax.set_yscale('log')
ax.grid(True)
ax.set_xlabel('Iteration Number')
ax.set_ylabel('Loss')
fig.suptitle(title)
ax.legend()
fig.savefig(title + '.png', bbox_inches='tight')