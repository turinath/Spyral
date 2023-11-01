# Spyral

Spyral is an analysis application for data from the Active Target Time Projection Chamber (AT-TPC). Spyral provides a flexible analysis pipeline, transforming the raw trace data into physical observables over several tunable steps. Sypral can process multiple data files in parallel, allowing for scalable performance over larger experiment datasets.

## Installation

### Download

To download the repository use `git clone https://github.com/turinath/Spyral.git`

To install the required packages it is recommended to create a virtual environment with either Anaconda or python/pip, detailed below.

### Anaconda

On the same directory as this package, run:

```[bash]
conda env create -f environment.yml
```

This creates an Anaconda environment with the name "spyral-env" with all of the necessary libraries and versions.

### Pip

Or if pip is prefered create a virtual environment using

```[bash]
python -m venv </some/path/to/your/new/environment>
```

Activate the environment using `source </some/path/to/your/new/environment/>/bin/activate`, then install all required dependencies using

```[bash]
pip install -r requirements.txt
```

All dependencies for Spyral will then be installed to your virtual environment

## Requirements

Python >= 3.10

if using Anaconda: Anaconda >= 4.10.1

Spyral aims to be cross platform and to support Linux, MacOS, and Windows. Currently Spyral has been tested and confirmed on MacOS and Ubuntu 22.04 Linux. Other platforms
are not guaranteed to work; if there is a problem please make an issue on the GitHub page, and it will be resolved as quickly as possible.

## Usage

### Configuration

User configuration parameters are passed through JSON files. Configuration files are passed at runtime to the script.

Configurations contain many parameters. These can be seen in the config.json example given with the repo. These parameters are grouped by the use case:

- Workspace parameters: These are file paths to either raw data, the workspace, or various AT-TPC pad data files.
- Run parameters: Run numbers over which the data should be processed, as well as indications of which types of analysis to run
- Detector parameters: detector conditions and configuration
- Trace parameters: parameters which are used in the peak identification and baseline removal analysis
- FRIB trace parameters: parameters used in the peak identification of FRIBDAQ signals (ion chamber, auxilary silicon, etc)
- Cross-talk parameters: parameters used in cross-talk removal, after peaks have been identified
- Clustering parameters: point cloud clustering parameters
- Estimation parameters: used to generate estimates of physical observables
- Solver parameters: used to control the physics solver

### Running

To use Spyral, run the main.py script located at the top level of the repository with the virtual environment activated. Example:

```[bash]
python main.py <your_config.json>
```

Replace `<your_config.json>` with the path to your configuration file.

### Performance

Spyral attempts to be as performant as possible while also being flexible enough to handle the broad sea of data that is generated by the AT-TPC. To that end, below are some useful tips
on extracting the most performance out of the application.

- Phase 1 is by far the most time consuming task by our benchmarks. Some of the bottleneck is the I/O on the raw traces; raw trace files range in size from 10 GB to 50 GB, and an event can be several MB on it's own. As such it is highly recommended to store the trace data on a SSD rather than an HDD. Additionally, when possible, it is also recommended to store the data on a local disk (i.e. SATA or NVME/PCIe). USB connected removable drives can represent serious bottlenecks to this part of the analysis.
- Phase 2 is entirely limited by the clustering algorithm chosen. As such there is little that can be done to improve the performance of this section. In general, Phase 2 is the second most time consuming task, but is still much faster than Phase 1 (typically a factor of 2).
- Phase 3 is very performant due to the relative simplicity of the analysis. In general Phase 3 should not be considered expensive
- For Phase 4 the story is complicated and will be described in more detail below.

### Phase 4

The final phase of the analysis involves using the equations of motion of a charged ion in a electromagnetic field to extract physics parameters. As it might sound, this isn't that straight forward. The simple approach is to fit ODE solutions to the data, but it can prove quite expensive to solve the ODE's the hundreds of times it takes to minimize per event. To bypass this expense, Spyral pre-calculates many of these ODE solutions and then interpolates on them to find a best fit. To make this even faster, Numba is used to just-in-time compile a lot of the interpolation code. As such, the first time you run phase 4, it might take a while because Spyral is generating the interpolation scheme. But after that it will be really fast!

An alternative approach, the Unscented Kalman Filter, also exists. But this approach is not sound yet; more testing and development needs to be done before this method is ready to be used in production.

### Parallel Processing

As was mentioned previously, Spyral is capable of running multiple data files in parallel. This is acheived through the python `multiprocessing` library. In the configuration file, there is a parameter named `n_processors`. The value of this parameter indicates to Spyral the *maximum* number of processors which can be spawned. Spyral will then inspect the data load that was submitted in the configuration and attempt to balance the load across the processors as equally as possible.

Some notes about parallel processing:

- The number of processors should not exceed the number of physical cores *MINUS* one in the system being used. Doing so could result in extreme slow down and potential unresponsive behavior.
- In general, it is best if the number of data files to be processed is evenly divisible by the number of processors. Otherwise, by necessity, the work load will be uneven across the processors.
- Spyral will sometimes run fewer processes than requested. This is usually in the case where the number of requested processors is greater than the number of files to be processed.

## Plotting

Spyral also bundles some helpful plotting tools for creating dataset histograms. The default numpy/scipy/matplotlib histogramming solution is not terribly useful for larger datasets. The tools included in spyral/plot can help generate histograms of large datasets as well as generate gates for use with various analyses. The plotter.py file contains an example of how to generate a particle ID gate and then apply it to a dataset. The example plotter can be run in two modes, gating and plotting. 

### Gating
To make a particle ID gate use

```[bash]
python plotter.py --gate <your_config.json>
```

where again you replace `<your_config.json>` with the path to your configuration file. You can draw a closed polygon around the particle group of interest. To save the gate, close the plot window and the gate will be automatically saved to the gate directory of your workspace with the name `pid_gate.json`. The PID file does need modified by the user before being used in Spyral. You will need to manually add the fields `Z` and `A` which are the proton and mass number of the particle associated with the group. The order in the JSON file does not matter, the fields simply need to have the correct name.

### Plot

To make a set of useful plots use

```[bash]
python plotter.py --plot <your_config.json>
```

This will produce some useful plots like the particle ID plot, the ion chamber energy, and the kinematic correlation of energy and angle. It will make a set with and without the particle ID gate applied. Note that this requires a real PID gate to be given by the configuration.