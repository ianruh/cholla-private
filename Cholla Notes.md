# Cholla Notes

**Running**

```
$ mpirun -n 4 ./cholla ./parameters.txt
```

*\* Each node on kodiak has four Tesla P100 GPUs*

Also, the modules I've been loading that work are:

```
intel/18.0.5
intel-mpi/2018.4
hdf5-parallel/1.8.16
cudatoolkit/10.1
```

\* *I changed to CUDA 10.1 because there was a bug in* `nvprof` *in CUDA 8.0 that would make it break when using mpi*.

**Restarting**

Just change the initial conditions to `Read_Grid` and add an `nfile` field that is equal to the number of the output files to start from:

```
...
init=Read_Grid
nfile=800      # 800 if starting at t=200 with t_out=0.25
...
```

\* *Needs to use the same number of processes as before.*

**Post Processing**

In the `ti` branch I modified the concatenation script she gave in `cholla/python_scripts/`. There is one for each 1D, 2D, and 3D: `cat_dset_1D.py`, `cat_dset_2D.py`, `cat_dset_3D.py`. You can run them with no arguments to get help text, but each one is essentially the same:

```
$ python cat_dset_2D.py
Usage: [command] [n_proc] [inDir] [outdir]
```

So to concatenate the raw output (i.e. `0.h5.0`, `0.h5.1`, `0.h5.2`, …) from a directory `raw_split/` and save the output (`0.h5`, `1.h5`, `2.h5`, …) in the directory `raw/` you would run (*assuming there were four mpi processes running*):

```
$ python cat_dset_2D.py 4 raw_split/ raw/
```

**Visualization Scripts**

For both the 1D and 2D script, you can get the help text just by running them without any arguments. The 2D script that I've been using is essentially the same as the original one you gave me, only with the Cholla code added and with taking the input from the command line arguments.

**Already Run Simulations**

1D: `/net/scratch4/iruh/1Druns/`

2D: `/net/scratch4/iruh/2Druns/`

You should have r/w access to both of them. All of the 1D runs should have their movie within that run's folder, and all of the 2D runs that I've made a movie for will have it in the folder.

\* *Within the 2D runs directory, the first level has all of the runs without Spitzer conductivity, then within the* `Var_Kappa/` *directory are all of the runs with Spitzer conductivity.*

**Useful Commands**

- `$ tail -n 5 stdout.txt`

  Print the last five lines of the given file.

- `$ head -n 5 stdout.txt`

  Print the first five lines of the given file.

**Some Git**

```
# List all the branches and show the currently checked out one
$ git branch

# Checkout a previous commit or another branch
$ git checkout [[branch] or [commit id]]

# Temporarily save all of the changes to your currently checked out version.
$ git stash
# List all of the stashes
$ git stash list
# Delete a stash
$ git stash drop [stash number]
# Apply a stash
$ git stash apply [stash number]
# Stashes carry through different branches, so you can stash changes from one
# branch, checkout another, then apply those changes to the second branch.

# Reset the working directory back to the most recent commit (any changes made
# will be lost)
$ git reset --hard
$ git reset --hard [commit id] # Reset to a specific commit
# Unstage all of the files currently staged. All changes will be preserved.
$ git reset
```

