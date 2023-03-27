import matplotlib.pyplot as plt

filename = 'log_RTX3060_optimalbatchsize128'
# Open the text file for reading
with open(filename + '.txt', 'r') as file:
    # Initialize empty lists to store data
    iterations = []
    losses = []
    # Iterate over each line in the file
    for line in file:
        # Split the line into words
        words = line.split()
        #print(words)
        # If the line contains the iteration number and loss value
        if words[2] == 'loss' and words[0] == 'iter':
            # Add the iteration number and loss value to their respective lists
            iterations.append(int(words[1].strip(':')))
            losses.append(float(words[3].strip(',')))


# the average loss value in the last 200 iterations of a training loop
last_200_losses = losses[-200:]
avg_loss = sum(last_200_losses) / len(last_200_losses)
print(f"Average loss in last 200 iterations: {avg_loss}")


# Find the index of the lowest loss
min_loss_idx = losses.index(min(losses))
# Add a red dot for the lowest loss
fig, ax = plt.subplots()
ax.plot(min_loss_idx, losses[min_loss_idx], 'ro', label=f"min = ({min_loss_idx}, {losses[min_loss_idx]})")

# Create a line plot with iteration number on the x-axis and loss value on the y-axis
ax.plot(iterations, losses)
ax.set_yscale('log')
ax.grid(True)
ax.set_xlabel('Iteration Number')
ax.set_ylabel('Loss')
fig.suptitle(filename)
ax.legend()
fig.savefig(filename + '.png', bbox_inches='tight')