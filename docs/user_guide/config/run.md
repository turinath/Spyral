# Run Configuration

The Run parameters control which data runs Spyral will analyze and what analysis will be performed. The default run parameters given in `config.json` are:

```json
"Run":
{
    "run_min": 17,
    "run_max": 17,
    "n_processes": 1,
    "phase_pointcloud": false,
    "phase_cluster": false,
    "phase_estimate": false,
    "phase_solve": false,
    "is_legacy": false
},
```

Note that by default run 17 is given and no phases are run, so nothing will actually happen.

A break down of each parameter:

## run_min

The minimum run number to be processed (inclusive). Spyral will skip any runs for which there is no available data.

## run_max

The maximum run number to be processed (inclusive). Spyral will skip any runs for which there is no available data.

## n_processes

The number of subprocesses to spawn. Each subprocess will be given a (roughly) equal load of data to process. The number of subprocesses, in general, should not exceed the total number of CPU cores in your system. If fewer runs are found than number of processes requested, Spyral will only use the number of processesed needed to run one run per process. Must be at least 1.

## phase_pointcloud

Select whether or not to run the point cloud generation phase, the first phase of the analysis. Set to `true` to run or `false` to skip.

## phase_cluster

Select whether or not to run the clustering phase, the second phase of the analysis. Set to `true` to run or `false` to skip.

## phase_estimate

Select whether or not to run the estimation phase, the third phase of the analysis. Set to `true` to run or `false` to skip.

## phase_solve

Select whether or not to run the solving phase, the fourth and final phase of the analysis. Set to `true` to run or `false` to skip.

## is_legacy

Controls whether or not data is analyzed in legacy mode. Legacy data is defined as data which was taken before the separation of auxilary detectors like the IC from the GET data acquisition into the FRIB acquisition. In this mode it is assumed that any data in CoBo 10 is auxilary data.
