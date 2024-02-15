# Spyral in Parallel

One of the other important performance features of Spyral is parallel processing of runs. In the [configuration](config/run.md), you can specify a range of runs to process as well as the number of processors. Each processor is an independent Python interpreter process, running it's own instance of Spyral. Below we'll describe a little bit about how this works and what some of the strenghts and limitations are.

## How it Works

The parent process (the one caused by you running Spyral in the terminal) examines your config, and then does some work to prepare for parallelization. First, it pre-generates all of the shared resources. This includes creating the workspace, interpolation meshes, and checking for the existence of data. It then gathers all of the runs you requested be processed and tries to balance the load on each processor, such that no one processor is being overloaded with data (this would make that processor take forever compared to the others, basically defeating the purpose of parallelizing Spyral).

The balancing works as follows: The runs are sorted by the size of their raw trace files. Runs are then given to processors in a snake method:

```txt
runs = [9, 0, 2, 1, 3, 4] in order of size

processor 0: <- 9 <- 4 => [9,4]
                |    ^
                V    |
processor 1: <- 0 <- 3 => [0,3]
                |    ^
                V    |
processor 2: <- 2 <- 1 => [2,1]
```

This way works generally well at load balancing (assuming all runs are equal in data complexity for analysis).

Each processor is given a queue it uses to communicate to the parent process. Messages are sent to the parent when the analysis status changes (new phase, increment progress, etc). The awesome [tqdm](https://github.com/tqdm/tqdm) library is used to monitor the progress of each child processor.

Each processor (including the parent) writes its own log file, which can be found in the `log/` directory of the workspace. This allows for error reporting and finer grained messaging, so that way when something goes wrong you can find out why!

## Optimizing Performance

To get the best performance, the number of runs to be analyzed should be evenly divisible by the number of processors. Otherwise, by necessity some of the processors will  generally *have* to run longer.  Also its best if runs are all of uniform size (but that's not usually within our control).

In general more processors is more better, but this isn't always the case. The first limit is the total number of cores in your machine. You need `n_processors`+1 physical cores, otherwise things are guaranteed to slow down. Another consideration is your CPU architecture. Intel i-series and Apple ARM use P-core and E-core packages. P-cores are performance cores which hit higher clock speeds and are optimized for expensive tasks. E-cores are efficiency cores, which run slower but are less power hungry, best suited to background tasks. Spyral is best suited to P-cores; as such you should use only P-cores as the number of processors available.

The final consideration is memory. This is mostly important for the solving phase. The interpolation mesh can be quite large (somtimes GBs) and needs to be stored in active memory *for each processor*. You need enough system memory (RAM) for each processor to completely load the mesh.

If you are using Spyral in a job environment (SLURM, etc), you can run with the `--no-term` option to disable printing to the terminal

```bash
python main.py --no-term local_config.json
```
