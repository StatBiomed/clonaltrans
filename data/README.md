# Data

We will put one larry dataset here as demo, including N & L, possibly also S

The file kinetic_array_correction_factor.txt shows the total number of cells over time (3) per mega-clone (10) per population (11). You can read and plot the data with:




loaded_arr = np.loadtxt("kinetics_array_correction_factor.txt")
  
load_original_arr = loaded_arr.reshape(loaded_arr.shape[0], loaded_arr.shape[1] // 11, 11)

fig, ax = plt.subplots(10,11, figsize=(22,15))

for i in range(10):
    for j in range(11):

        ax[i][j].plot([3,10,17],load_original_arr[i,:,j])


fig.subplots_adjust(hspace=0.5)
fig.subplots_adjust(wspace=0.5)



Note: this array only has barcoded cells. We also have non barcoded cells that we could use as background, but I haven't figured out yet what's the proportion of non barcoded cells that was sequenced. For the moment, please use the sum of megaclones as background. This is anyway a test, because more data is coming soon.


Similarly, I will update S as soon as I figure out the actual proportions. 





