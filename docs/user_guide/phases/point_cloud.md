# The First Phase: Point Clouds

The first phase of the analysis is the transformation of raw AT-TPC digitized waveforms (often called traces) into a point cloud geometry. There are four steps to this part of the analysis:

1. Removal of the trace baselines
2. Identifying signal peaks in the traces
3. Consolidation of the signals into point clouds
4. Identifying associated auxilary detectors and applying corrections

Below we'll walk through each of these steps

## Baseline Removal

Each trace has some DC offset, that is a shift from 0 on the ADC scale of the trace. To correct for this, a Fourier analysis techinque is employed. This technique was first used in AT-TPC analysis by J. Bradt in his [thesis work](https://d.lib.msu.edu/etd/4837?q=Bradt%20at-tpc). Very breifly we will go over some of the details here. A Fourier transform can be written as

$$
    \hat{f}(\xi) = \mathfrak{F}(f(x)) = \int^\infty_{-\infty} f(x)e^{-2\pi i \xi x} dx
$$

This transformation transforms a function to the frequency domain; however the main useful tool here what is known as the Convolution Theorem, which states that for two functions $f(x)$ and $g(x)$ with Fourier transforms $\hat{f}(\xi)$ and $\hat{g}(\xi)l$ the convolution $\{f * g\}(x)$ can be expressed as the inverse Fourier transform of the prodcut of the Fourier transforms; that is

$$
    \{f * g\}(x) = \mathfrak{F}^{-1}(\hat{f}(\xi)  \hat{g}(\xi))
$$

This is really powerful! It means calculating a covolution is as simple as multiplying the Fourier transforms of the functions and taking the inverse Fourier transform of that product. Numpy provides the libraries for calculating Fourier transforms, and we can speed them up using the Rocket-FFT library and Numba.

We still haven't exactly explained why this is useful. Removing a baseline is kind of like removing an average, or applying a low-pass filter. In this way, we basically want to convolve a window or averaging range with our signal to extract a smooth baseline to remove (remember that the baseline is noisy as well). Such a function is basically a step

$$
    g(x) = \begin{cases} 1\text{, } -\frac{1}{2} < x < \frac{1}{2} \\ 0\text{, elsewhere} \end{cases}
$$

Conviently for us, the Fourier transform of a step has a simple description

$$
    \hat{g}(\xi) = \text{sinc}(\xi) = \frac{\sin(\pi \xi)}{\pi \xi}
$$

We introduce an aditional scaling parameter $a$ so that $\hat{g}(\xi) = \text{sinc}(\xi/a)$ and the size of the baseline window can be controlled. This means that all we need to do is take the Fourier transform of our trace, multiply it by our $\hat{g}$, and then take the inverse fourier transform of that! We can even do this in one shot for the GET traces, correcting all of the baselines at once using the 2-D Fourier transform algorithms of Numpy.

There are some small details that can be interesting.Before taking the Fourier transform of the trace, we need to remove singals from the trace, otherwise they will upset the baseline, moving the average up. This is done removing any time-buckets which have a value $\ge 1.5 \sigma$ away from the mean amplitude of the trace where $\sigma$ is the standard deviation of the trace amplitude.

This code is entirely contained in `spyral/trace/get_event.py` in the `preprocess_traces` function.

## Identifying Signal Peaks

Identifying peaks in spectra (traces) can be a tedious, delicate, and time consuming task. Thankfully, it is a common enough problem that someone already solved it! Scipy has a `signal` library that contains the `find_peaks` function. `find_peaks` takes a lot of different arguments that can greatly impact the quality of the peak finding. Chekcout our [configuration](../config/traces.md) description to see which parameters are exposed and what they do.

Once peaks are identified, centroids, amplitudes, integrals are extracted.

This code is contained in `spyral/trace/get_trace.py` and `spyral/trace/get_event.py`.

## Consolidation into Point Clouds

Once peaks have been identified, it is time to convert from signals into geometry. Each signal represents a measurement of the flight path of a particle through the AT-TPC (in principle anyways, there's a lot of noise). Coordinates in the AT-TPC pad plane (X-Y) are taken using the pad ID in which the signal occurred and looking up the pad position from a CSV file (again see the [config](../config/workspace.md) docs). The beam-axis coordinate (Z) is extracted by calibrating the centroid location of the peak, which originally is in units of GET time-buckets, to millimeters. This is done using two aboslute reference points: the window and the micromegas. The time bucket of the window and micromegas can be extracted by examining events with long trajectories such as events where the beam reacted with the window. The calibration is then

$$
    z_{signal} = \frac{t_{window} - t_{signal}}{t_{window} - t_{micromegas}} l_{attpc}
$$

where $l_{attpc}$ is the length of the AT-TPC in millimeters.

## Auxilary Detectors and Corrections

The AT-TPC has at minimum one auxilary detector, the upstream ion chamber. The ion chamber (IC) is used to generate the AT-TPC trigger (in conjunction with the mesh) and identify beam species, and is recorded using the FRIBDAQ framework. Its data is analyzed very similarly to the GET data; baselines are removed and signals are extracted. Users can specify the maximum allowed ion chamber multiplicty to reject data with has an inconsistent trigger. Usually the AT-TPC also has a silicon detector downstream within the hole of the pad plane. This is useful for correcting the time of an event which triggered from the wrong IC signal by examining coincidences between the silicon and IC; only IC events which do not have a silicon coincidence are valid triggers. See the code in `spyral/traces/frib_traces.py` and `spyral/traces/frib_event.py` for details.

Additionally, it has been found that in some cases the AT-TPC needs a correction for distortions to the electric field. The correction is calculated using [Garfield++](https://gitlab.cern.ch/garfield/garfieldpp) externally from Spyral. Spyral takes in a correction file containing initial electron positions and final drift positions and times. Spyral generates an interpolation mesh to calculate corrections for each point. The mesh is calculated and stored in a Numpy format file (`.npy`). It is not clear how impactful this correction is anymore; it was seemingly most important early on in the AT-TPC development. The associated code can be found in `spyral/correction`. This correction is optional, and can be turned off and on using the configuration.

Point clouds are also sorted in z for ease of use later.

## Legacy Data

In the past, the AT-TPC was not run with a split DAQ. That is, all signals were recorded using the GET data aquisition, including auxilary detectors like the ion chamber. In the configuration, one can indicate to Spyral that legacy data is being analyzed (see [here](../config/run.md)). If this is turned on Spyral assumes that any data in CoBo 10 is exclusively from auxilary detectors. Legacy mappings and corrections are included in the `etc` directory. Legacy IC data is analyzed for peaks using the `FRIB` parameters even though the data is acquired using the GET DAQ (see [here](../config/traces.md)). Legacy analysis is an advanced feature, and requires some experience with using Spyral. This phase is the only phase significantly impacted by legacy analysis.

## Final Thoughts

The first phase is very intense and represents a major data transformation. Typically, the trace data is somewhere between 10-50 GB per run, and the output of the point cloud phase is something like 250 MB to 1 GB. This is an enormous reduction of data, and as such is usually the slowest phase, taking anywhere from 10 minutes to an hour depending on the hardware being used. The bottleneck is typically I/O speed; reading in so much data is a serious issue depending on the type of storage used to hold the traces. In general, if the traces are stored on a network drive which doesn't have hardline 10 Gb connection, this phase will be slow. HDD drives can also be a slow down, or older USB connected external drives. The general recommendation is to move data to a fast local SSD for analysis when possible.
