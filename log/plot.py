import matplotlib.pyplot as plt

# Open the text file for reading
with open('log_RTX3060ti_kaparthy_bs2.txt', 'r') as file:
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

# do the same thing for original model
#TODO

# the average loss value in the last 200 iterations of a training loop
last_200_losses = losses[-200:]
avg_loss = sum(last_200_losses) / len(last_200_losses)
print(f"Average loss in last 200 iterations: {avg_loss}")


## plot 
#TODO
#have loss for original model as well


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
fig.suptitle('RTX3060ti-Kaparthy')
ax.legend()
fig.savefig("log_RTX3060ti_kaparthy_bs2.png", bbox_inches="tight")