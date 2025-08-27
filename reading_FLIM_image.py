
######################################################################

# "Simple" script to analyze FLIM data
# m.wehrens@uva.nl, 2024-10-21

# This script analyzes data for a "mini project" (part of advanced microscopy course),
# for which lifetime measurements were conducted in cells expression a calcium sensor
# of which the lifetime changes in response to binding calcium.

# The aim of the project is to test the sensitivity of the sensor to calcium changes
# in conditions with low calcium concentrations.

# Thus, several samples will be measured in conditions that have varying concentration
# of (intracellular) calcium, and we need to quantify the lifetime of the sensor in each
# of those conditions.

# The approach to do this will be to use a phasor plot. The samples will each have 
# multiple cells in the field of view, and we'd like to quantify the fluorescent
# lifetime in each of those cells separately.

# The goal of this current script is to 
# - read in ptu data.
# - convert this to an intensity plot.
# - have a simple treshold function determine a mask to identify cells.
# - allow users to sophisticate this mask using FIJI.
# - extract the intensity traces for each cell
# - calculate the phasor coordinates for each cell

# In the current version, the script will analyze 1 specific sample,
# but this can later be generalized/automated to handle multiple
# samples.

######################################################################
# External library that is used in this script to read .ptu files

# Downloaded libraries from others
# PTU file reader, from https://github.com/cgohlke/ptufile
#from ptufile import PtuFile as ptf 
import ptufile as pf
    # Notes about the format; load using either:
    # ptu = pf.PtuFile('filename.ptu')
    # or
    # ptu = pf.imread('filename.ptu')
    # This results in in array with the following information:
    # (T, Y, X, C, H), where T is time, Y is the y-axis, X is the x-axis, C is the channel, 
    # and H are the "histogram bins", ie the time bins.
    # See https://github.com/cgohlke/ptufile/blob/main/ptufile/ptufile.py, decode_image
    # Importantly, some information regarding the ptu file can also be read out, using
    # ptu.frequency #, the frequency that was used to acquire the data
    # ptu.global_resolution #, the time window the measurement covers
    # ptu.number_bins_in_period #, the amount of bins the time window is divided in
    # 
    # Note that ptufile is made by Christoph Gohlke, who is also working on PhasorPy
    
######################################################################
# Import more standard python libraries

# Standard libraries
import os
import matplotlib.pyplot as plt # plotting
import seaborn as sns # plotting using dataframes 
import numpy as np # numerical calculations

# Some extra convenient things
from matplotlib import patches

# Image reading and writing
import skimage.io as skio

# Image processing libraries
from scipy.ndimage import binary_erosion, binary_dilation, binary_closing, binary_opening
from skimage.morphology import erosion, dilation, square, disk
from skimage.measure import label, regionprops


# Working with dataframes
import pandas as pd

import cv2

# Currently not used
# from scipy.optimize import curve_fit


# Bang Wong colorblind-friendly color scheme (https://www.nature.com/articles/nmeth.1618)
color_palette = [
    "#E69F00",  # Orange
    "#56B4E9",  # Sky Blue
    "#009E73",  # Bluish Green
    "#F0E442",  # Yellow
    "#0072B2",  # Blue
    "#D55E00",  # Vermillion
    "#CC79A7",  # Reddish Purple
    "#000000"   # Black
]
# plt.figure(); plt.scatter(list(range(len(color_palette))), list(range(len(color_palette))), color=color_palette); plt.show()

cm_to_inch=1/2.54 # conversion cm to inch (convenient for plotting)

######################################################################

# Define the path to the ptu file that needs to be analyzed
# MYFILEPATH = '/Users/m.wehrens/Data_UVA/2024_08_Sebastian_cAMPING_HeLa/DataSeb_20240801/TL2_FLIM_cAMPinG1_Iso_SATURATED.sptw/TL2_FLIM_cAMPinG1_Iso_SATURATED_t1.ptu'
MYFILEPATH = './example_data/cytoplasm_A2_08.ptu'
CHANNEL_OF_INTEREST = 0 # for now, we'll only analyze 1 channel

OUTPUT_DIRECTORY = '/Users/m.wehrens/Data_UVA/2024_teaching/2024-10-miniproject/analysis/'
# create the output directory if it doesn't exist
os.makedirs(OUTPUT_DIRECTORY, exist_ok=True)

######################################################################

# Read in the file of interest
ptu_im = pf.imread(MYFILEPATH, dtime=0)[0,:,:,:,:] # reading it this way ensures that the number of bins is correct
    # [0,:,:,:,:] throws out the first dimensions, which we don't need
ptu = pf.PtuFile(MYFILEPATH) # reading it this way provides additional info (see below)

# Check some important features of the data
ptu.shape # simply dimensions of the data, (T, Y, X, C, H)
ptu_im.shape # same as above, but without the first dimension (Y, X, C, H)
ptu.frequency # the frequency that was used to acquire the data
ptu.global_resolution # the time window the measurement covers
ptu.number_bins_in_period # the amount of bins the time window is divided in

# now create an intensity image by summing over the time dimension
img_intensity = np.sum(ptu_im[:, :, CHANNEL_OF_INTEREST, :], axis=2) 
img_intensity.shape

# Show this image in a plot
fig, ax = plt.subplots(1, 1, figsize=(10*cm_to_inch , 10*cm_to_inch))
_ = ax.imshow(img_intensity, cmap='gray')
# plt.show()
plt.tight_layout()
fig.savefig(OUTPUT_DIRECTORY+'intensity_image.pdf', dpi=300, bbox_inches='tight')
plt.close(fig)

# Rescale the image and save the image as a tif using skimage.io
# Also adjusts the image format for working in cv2
img_intensity_rescaled = np.array(img_intensity/np.max(img_intensity)*(2**16-1), dtype=np.uint16)
skio.imsave(OUTPUT_DIRECTORY+'intensity_image.tif', img_intensity_rescaled)

######################################################################
# Now perform simple thresholding to identify the cells

# Calculate an automatic treshold
value_treshold_Otsu, img_intensity_mask = \
    cv2.threshold(img_intensity_rescaled, 0, np.max(img_intensity_rescaled), cv2.THRESH_BINARY + cv2.THRESH_OTSU)

# show histogram of img_intensity_mask
fig, ax = plt.subplots(1, 1, figsize=(10*cm_to_inch , 10*cm_to_inch))
_ = ax.hist(img_intensity_rescaled.flatten(), bins=255)
# set y-axis to log scale
ax.set_yscale('log')
plt.show()

# show the mask
fig, ax = plt.subplots(1, 1, figsize=(10*cm_to_inch , 10*cm_to_inch))
_ = ax.imshow(img_intensity_mask, cmap='gray')
plt.show()
#plt.tight_layout()
#fig.savefig(OUTPUT_DIRECTORY+'intensity_mask.pdf', dpi=300, bbox_inches='tight')

def improve_mask(mask):
    
    # close the mask using binary closing from scipy    
    mask_closed = binary_closing(mask, disk(5))
    
    # remove small objects:
    
    # first determine regions
    mask_labeled = label(mask_closed)
    # then determine properties of the regions
    regions = regionprops(mask_labeled)
    
    # now only keep regions with size >100px
    mask_filtered = np.zeros_like(mask_closed)
    for region in regions:
        if region.area > 100:
            mask_filtered[region.coords[:,0], region.coords[:,1]] = 1
    
    # return the result    
    return(mask_filtered)

# improve the mask
img_intensity_mask_impr = improve_mask(img_intensity_mask)
# show the mask
fig, ax = plt.subplots(1, 1, figsize=(10*cm_to_inch , 10*cm_to_inch))
_ = ax.imshow(img_intensity_mask_impr, cmap='gray')
plt.show()

# save the segmentation as a tif
skio.imsave(OUTPUT_DIRECTORY+'intensity_image_segmented.tif', np.array(img_intensity_mask_impr*255, dtype=np.uint8))


######################################################################
# Now import the maks and extract signals for each cell
# Note that you can change the mask in FIJI if you want to improve it

# Load the mask from tif file
img_user_mask = skio.imread(OUTPUT_DIRECTORY+'intensity_image_segmented.tif')/255

# Now create regions from the mask
img_user_labeled = label(img_user_mask)

# Now extract the intensity traces for each cell
# (Note that in the regions, cells are labeled from 1 to n, but in indexing, they are labeled from 0 to n-1)
list_of_signals = [[]]*np.max(img_user_labeled)
for cell_idx in range(1, (np.max(img_user_labeled)+1)):
    
    # select the pixels from the cells
    current_cell_coordinates = np.where(img_user_labeled==cell_idx) 
    ptu_cellpixels = ptu_im[current_cell_coordinates[0], current_cell_coordinates[1], :, :]
        # ptu_cellpixels.shape
        # e.g. (1339, 1, 132) = pixels, channel, time
    
    # now calculte the sum of the signal (photons) over the cell
    list_of_signals[cell_idx-1] = np.sum(ptu_cellpixels[:, CHANNEL_OF_INTEREST, :], axis=0)

# create a time axis too
array_time = np.linspace(0, ptu.global_resolution, ptu.number_bins_in_period)
array_time_ns = array_time*1e9

# Determine the location of the peak, arbitrarily for the first cell
t_maxsignal_cell1_idx = np.argmax(list_of_signals[0])
t_maxsignal_cell1 = array_time[t_maxsignal_cell1_idx]

# Create a dataframe with the signals (for plotting)
# create  a list of dataframes
list_dfs = [pd.DataFrame({'time':array_time, 'time_ns':array_time_ns, 'signal':list_of_signals[idx], 'cell':[idx+1]*len(list_of_signals[idx])}) for idx in range(0, np.max(img_user_labeled))]
# create a single dataframe
df_signals = pd.concat(list_dfs, axis=0)

# Now plot the signals per cell using seaborn
fig, ax = plt.subplots(1, 1, figsize=(10*cm_to_inch , 10*cm_to_inch))
sns.lineplot(data=df_signals, x='time_ns', y='signal', hue='cell', ax=ax)
# Also show the determine time of the peak
_ = ax.axvline(x=t_maxsignal_cell1*1e9, color='k', linestyle='-', label='max sig')
# add vertical line at the peak
plt.show()

######################################################################
# Calculate the semi-circle for a single-exponential for "all" tau values
# This part of the code can be done more elegant

T = ptu.global_resolution
def phasor_singletau(tau, T=T):
     t=np.arange(0, T+T/1000, T/1000)
     Y = np.exp(-t/tau)
     # plt.plot(t,Y); plt.show()
     phasor=np.sum(Y * np.exp(1j * (2*np.pi)/T * t)) / np.sum(Y)
     return [phasor.real , phasor.imag]

mytaus = np.logspace(np.log10(0.1), np.log10(1000), num=100)*1e-9 # going from 0.1 to 1000 ns
phasor_circle = np.array([phasor_singletau(tau) for tau in mytaus]) # calculate the phasor for each tau

######################################################################
# Now create polar coordinates for each cell

list_phasorX = np.array(np.empty(len(list_of_signals)))
list_phasorY = np.array(np.empty(len(list_of_signals)))
list_tau_intensity = np.array(np.empty(len(list_of_signals)))
for cell_idx in range(len(list_of_signals)):

    # First, we need to calculate the phasor coordinates for each cell
    the_signal = list_of_signals[cell_idx]

    # location of the peak
    t_maxsignal_currentcell_idx = np.argmax(the_signal)
    t_shifted= array_time - array_time[t_maxsignal_currentcell_idx]

    # Total signal
    total_signal_rec = np.sum(the_signal)
    # Total time
    T = ptu.global_resolution
    # Int ( I(t) * exp(-iwt) / Int (I(t))
    FT_trace = np.sum(the_signal * np.exp(1j * (2*np.pi) * t_shifted/T))/total_signal_rec # DT drops out

    # Convert to phasor
    list_phasorX[cell_idx] = FT_trace.real
    list_phasorY[cell_idx] = FT_trace.imag
    
    # Estimating the lifetime using the intensity weighed method
    area_signal   = np.sum(the_signal)
    area_signal_t = np.sum(the_signal * t_shifted)
    list_tau_intensity[cell_idx] = area_signal_t/area_signal
    
## Sanity check using flimtools
if False:
    # FLIM tools, from https://github.com/jayunruh/FLIM_tools, 
    # Download to custom location and import from that location
    sys.path.append('/Users/m.wehrens/Documents/git_repos/_UVA/FLIM_playground/FLIM_tools') # Add this directory to sys.path
    import flimtools  # or specific functions/classes

    # values are approximately similar to flimtools, i think due to slightly different handling of the time axis
    flimtools_estimate = flimtools.calcGS(the_signal, (t_maxsignal_currentcell_idx)/(len(the_signal)-1)*360, 1)
    print('FLIM tools calculation vs my calculation'+str(cell_idx)+':')
    print(flimtools_estimate)
    print([list_phasorX[cell_idx], list_phasorY[cell_idx]])


# Show the locations in a plot
fig, ax = plt.subplots(1, 1, figsize=(10*cm_to_inch , 5*cm_to_inch))
_ = ax.scatter(list_phasorX, list_phasorY)
# Add the semicircle defined above by mytaus and phasor_circle
_ = ax.plot(phasor_circle[:,0], phasor_circle[:,1], color='k', linestyle='--')
# Set axis equal
_ = ax.axis('equal')
# Label axes
ax.set_xlabel('G'); ax.set_ylabel('S');
plt.tight_layout()
plt.show()

# Also calculate the life time
tau_fromphasor = (list_phasorY/list_phasorX)/(2*np.pi/T)
tau_fromphasor*1e9 # in ns

# Compare the two methods
list_tau_intensity
tau_fromphasor