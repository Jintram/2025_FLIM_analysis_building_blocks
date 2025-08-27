

# Installation requirements

Set up the following environment to run this script:

```
conda create -n 2025_FLIM2 -c conda-forge numpy matplotlib seaborn pandas scikit-image scipy opencv ipython -y 
conda activate 2025_FLIM2
python -m pip install -U "ptufile[all]"
```

# Reading ptu data

The file `reading_FLIM_image.py` is an example script that shows how to read a ptu file. (It's based on a previous example script I wrote for a course.)

To correctly process the image, you'll need to know technical details about the measurement, such as the measurement period and number of bins that correspond to the actual measurement.
These can be read out with the `ptufile` package by Christoph Gohlke, see code snippet below. I also recommend this package for reading in the ptufiles to python arrays.

Importantly, when importing data with `ptufile`, it's convenient to use the `dtime=0` setting to make sure the array size is consistent with the number of bins.

Code snippet:
```
import ptufile as pf

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

For more information, see the `reading_FLIM_image.py` example script (from which this code snippet is lifted).
An example image is provided in the `example_data` directory.


# A note about "flimtools"

Sometimes people in our lab have used the "flimtools" package (https://github.com/jayunruh/FLIM_tools) to calculate G, S coordinates for a FLIM decay curve. E.g. by using the function:

```
def calcGS(profile,pshift,mscale,harmonic=1):
    plen=len(profile)
    pshiftscale=pshift*plen/360.0
    return [(mscale*np.cos(2.0*np.pi*harmonic*(np.arange(plen)-pshiftscale)/plen)*profile).sum()/profile.sum(),
            (mscale*np.sin(2.0*np.pi*harmonic*(np.arange(plen)-pshiftscale)/plen)*profile).sum()/profile.sum()]
```

The "pshift" argument is very important, it shifts the curve in time to align the peak to t=0,
which is crucial to a proper G, S coordinate calcualation.
Counter-intuitively, the pshift argument has units of degrees, going from 0-360 degrees,
where 0 corresponds to t=0, and 360 to the total period.
I'm not sure this is documented somewhere, but it can be seen from the above code.

## Notes to self

*(Please ignore this section.)*

<font color=grey>

Some convenient files:
- /Users/m.wehrens/Documents/PROJECTS/_Documentation_and_theory/tex_FLIM/Some_math_behind_FLIM.pdf
</font>