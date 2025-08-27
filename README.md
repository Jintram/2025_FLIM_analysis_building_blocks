

# Installation requirements

Set up the following environment to run this script:

```
conda create -n 2025_FLIM2 numpy matplotlib seaborn pandas scikit-image scipy opencv -y
conda activate 2025_FLIM2
python -m pip install -U "ptufile[all]"
```

# Reading ptu data

The file `reading_FLIM_image.py` is an example script that shows how to read a ptu file.

To correctly process the image, you'll need to know technical details about the measurement, such as the measurement period and number of bins.
These can be read out with the `ptufile` package by Christoph Gohlke, see code snippet below.
It's also convenient to use the `dtime=0` setting to make sure the array size is consistent with the number of bins.

```
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
```

For more information, see the `reading_FLIM_image.py` example script.
An example image is provided in the `example_data` directory.