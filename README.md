# Manifold based Dynamic Free Breathing Pulmonary UTE MR Image Reconstruction and Retrospective Motion Compensation for Neonatal Subjects with Bulk Motion

## Background
Dynamic lung MRI in neonates is challenging because rapid breathing and bulk motion make high-resolution image reconstruction severely ill-posed. Manifold-based methods that jointly estimate image and motion show promise for high-quality pulmonary imaging in this vulnerable population.

## Objective
To further develop an existing manifold-based reconstruction method and apply it to time resolved pulmonary imaging in a neonatal cohort for mitigation of respiratory and bulk motion artifacts.

## Layout


|Directory|File|Function|
|:--------|:---|:-------|
|`configs/`|`nicu.yaml`|Parameter file for the code run|
|`dataloader/`|`raw_io.py`|Loads and reads `MRI_Raw.h5`/`.npy` files containing the raw k-space `(n_coils, n_views, n_samples)`, trajectory `(n_views, n_samples, dim)` and density compensation weights`(n_views, n_samples)`|
||`coils.py`|Performs coil selection, PCA coil compression, computes sensitivity maps|
||`navigator.py`|Extract respiratory/cardiac/bulk latents from DC navigator|
||`binning.py`|Bins the data into frames based on motion states of latents|
||`operators.py`| Performs per-state NUFFT, weighting and computes coil maps per scale|
|`model/`|`generator.py`|Generates motion fields based on the latent values using a CNN and upscales to $64^3/128^3/256^3/320^3$ based on the resolution.
||`spatial_transformer.py`|Deform the template image using the motion fields geenerated by the generator.
||`regularizers.py`|Regularizers for the loss function.
|`executor/`|`config.py`|Validate the configuration file parameters and converts to yaml from JSON format (if needed)|
||`trainer.py`|Training loops and generator alignment|
||`pipeline.py`|The multi-resolution stages|
||`results.py`|`Results.h5` + NIfTI writing
||`cli.py`|`python -m executor`|
|`evaluation/`|`results.py`|Loads a run from `Results.h5`|
||`render.py`|Render motion states to NIfTI|
|`gui/`|`moco_viewer.py`|Interactive Tkinter viewer for `Results.h5`|
|`utils/`|`device.py`|Set the device|
||`log.py`|Set the logger to log info| 
||`plotting.py`|Helper functions to plot necessary stats|
||`reproducibility.py`|Set seeds to generate reproducible results.|
|`scripts/`|`build_container.sge`|Job script to build the image|
||`mocostorm-exec`|Runs necessary commands inside the container to bind required directories|
||`run_recon.sge`|Run the reconstruction|
|`container/`|`mocostorm.def`|Apptainer definition file|



## Setup

> All commands mentioned here including the `.sge` script files are for SGE based HPC cluster and tested on UIowa's Argon HPC cluster. If you're using a SLURM based cluster, take help from AI to convert the commands.

The code runs inside an Apptainer image, so nothing has to be installed on the cluster. The image contains the necessary packages from `environment.yml` (Python 3.9, torch 1.12.0+cu116, sigpy 0.1.26, cupy-cuda114 10.6) on the CUDA 11.4 runtime. However, make sure the system or the HPC compute node has the Apptainer and proper NVIDIA GPU driver installed.

Build the image during setup. You need to rebuild it only when `environment.yml` or `container/mocostorm.def` changes:

```bash
# Create a directory to store the logs in a scratch space
mkdir -p /nfsscratch/Users/$USER/mocostorm_logs
qsub scripts/build_container.sge            
```
It should save a `mocostorm.sif` file in the `containers/` directory. By default, it saves the container in `~/containers/` directory but you can always save it inside other directory using:

```bash
qsub scripts/build_container.sge /path/to/other.sif
```

If you don't want to use Apptainer (recommended way), you can use `conda env create -f environment.yml` to build the same environment. However, you may need to deal with installing earlier version of CUDA (11.4) and CuDNN (8.2.4) and resolve dependency conflicts. 

## Running a reconstruction

1. Write a config or use the provided one for test run. Make sure to add the required paths.
2. Submit a job on the node using the following command:
   
   ```bash
   # Create a directory to store the logs
   mkdir -p /nfsscratch/Users/$USER/mocostorm_logs

   # Use the provided config to run the code
   qsub scripts/run_recon.sge configs/nicu.yaml
   
   # However, any parameter can be overridden
   qsub scripts/run_recon.sge configs/nicu.yaml --set training.epochs.64=50 --run-name ep50
   ```

The job script asks for one A100 on the queue set in the `run_recon.sge` file. The run needs a GPU with about 40 GB free memory.

## Outputs

Each run is save in a different folder, `<data.output_dir>/<run_name or timestamp>/`
(default `<data_dir>/mocostorm_runs/...`).

| File | Content |
|---|---|
| `config.yaml` | The fully resolved config of this run so you don't have to remember the best set of parameters.|
| `recon.log` | Logs the error and other info needed for debugging or status checking |
| `Results.h5` | Stores the generator weights, template image and latents to view the dynamic image after the model is trained. |
| `X_<state>.nii.gz` | Magnitude image at the state nearest the median latent |
| `snapshots/` | Coil images and maps, latents, template slices during training |



## Viewer

`gui/moco_viewer.py` is the viewer for visualizing the dynamics captured by the latents. It runs inside the same Apptainer image, so no separate environment is needed. However, it needs a display and a CUDA GPU. So, start it from an interactive GPU session with X forwarding enabled (e.g. `FastX`, or `ssh -X` followed by `qlogin` with a GPU request).

From the repository root:

```bash
scripts/mocostorm-exec python -m gui.moco_viewer
```

`mocostorm-exec` adds `--nv` on a GPU node, and Apptainer passes `DISPLAY` and your home directory (with `~/.Xauthority`) through to the container. Once the viewer opens, you need to load the `Results.h5` file and allow it sometime to load all the saved weights and files. 


## Reproducibility
`runtime.seed` (default 0) seeds Python, NumPy and torch. It fixes the random choices: state order, coil subsets, PCA subsampling. However, GPU kernels are not bit-deterministic, so repeated runs agree closely but not bitwise.


## Reference
If you find this project useful for your work please cite:

```
[Will be added later]
```