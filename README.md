# Automated processing of MRI in neuro-oncology

This container provides easy to use access to our automated processing tool (HD-GLIO-AUTO) for brain tumor MRI scans. HD-GLIO-AUTO is the result of a joint project between the Department of Neuroradiology at the Heidelberg University Hospital, Germany and the Division of Medical Image Computing at the German Cancer Research Center (DKFZ) Heidelberg, Germany. If you are using HD-GLIO-AUTO please cite the following three publications:

* Kickingereder P, Isensee F, Tursunova I, Petersen J, Neuberger U, Bonekamp D, Brugnara G, Schell M, Kessler T, Foltyn M, Harting I, Sahm F, Prager M, Nowosielski M, Wick A, Nolden M, Radbruch A, Debus J, Schlemmer HP, Heiland S, Platten M, von Deimling A, van den Bent MJ, Gorlia T, Wick W, Bendszus M, Maier-Hein KH. Automated quantitative tumour response assessment of MRI in neuro-oncology with artificial neural networks: a multicentre, retrospective study. Lancet Oncol. 2019 May;20(5):728-740. (https://doi.org/10.1016/S1470-2045(19)30098-1)
* Isensee F, Schell M, Tursunova I, Brugnara G, Bonekamp D, Neuberger U, Wick A, Schlemmer HP, Heiland S, Wick W, Bendszus M, Maier-Hein KH, Kickingereder P. Automated brain extraction of multi-sequence MRI using artificial neural networks. Hum Brain Mapp. 2019; 1–13. (https://doi.org/10.1002/hbm.24750)
* Isensee F, Petersen J, Kohl SAA, Jaeger PF, Maier-Hein KH. nnU-Net: Breaking the Spell on Successful Medical Image Segmentation. arXiv preprint 2019 arXiv:1904.08128. (https://arxiv.org/abs/1904.08128)

The HD-GLIO-AUTO container requires the following MRI sequences from a brain tumor patient as input (with MRI sequences either in DICOM or NIfTI format):

* T1-weighted sequence before contrast-agent administration (T1-w) acquired as 2D with axial orientation (e.g. TSE) or as 3D (e.g. MPRAGE)
* T1-weighted sequence after contrast-agent administration (cT1-w) acquired as 2D with axial orientation (e.g. TSE) or as 3D (e.g. MPRAGE)
* T2-weighted sequence (T2-w) acquired as 2D 
* Fluid attenuated inversion recovery (FLAIR) sequence acquired as 2D with axial orientation (e.g. TSE). A 3D acquisition (e.g. 3D TSE/FSE) may work as well.

(these specifications are in line with the consensus recommendations for a standardized brain tumor imaging protocol in clinical trials - see Ellingson et al. Neuro Oncol. 2015 Sep;17(9):1188-98 - www.ncbi.nlm.nih.gov/pubmed/26250565)

After processing a given MRI examination with HD-GLIO-AUTO the following output files are automatically generated:

* Brain-extracted and co-registered MRI sequences (as nii.gz files)
* Brain extraction mask (as nii.gz file)
* Tumor segmentation mask (as nii.gz file)
* Tumor volume statistics (as txt file)
* (optional) z-score normalized cT1 - T1 subtraction (as .nii.gz file)

The output nii.gz files and segmentation masks can be easily displayed with an appropriate image viewer (e.g. [ITK-SNAP](http://www.itksnap.org/) or [MITK](http://www.mitk.org/)).

Please follow the instructions below for installation and usage of HD-GLIO-AUTO.

## Prerequisites

Using the HD-GLIO-AUTO container requires a GPU (no CPU support is provided). You need to install nvidia-docker version 2.0 to use this image. For instructions on how to install it follow [this link](https://github.com/nvidia/nvidia-docker/wiki/Installation-(version-2.0)). If you don't make nvidia your default docker runtime, you will need to replace `docker run` with `nvidia-docker run` in the next section. You can get the image by running

    docker pull jenspetersen/hd-glio-auto

The repository may be updated in the future.

## Usage

Set up a folder with your inputs. You will need these four:

    T1 (native T1)
    CT1 (contrast-enhanced T1)
    T2
    FLAIR

For each of these inputs there must be a folder with the DICOMs (e.g. the folder named CT1 or T2) OR a NIfTI file (the file named e.g. CT1.nii or T2.nii.gz). If multiple possible inputs are present the order of preference is `.nii.gz` > `.nii` > DICOMs.
The input types can be freely combined, the contents of your input folder could look like this, for example:

    T1.nii.gz
    CT1 (folder)
    T2.nii
    FLAIR.nii.gz

**NOTE**: NIfTI files with multiple temporal volumes (e.g. 4D cT1-w sequences) are not supported (however can be splitted upfront into the individual temporal volumes using [FSL´s fslsplit](https://fsl.fmrib.ox.ac.uk/fsl/fslwiki/Fslutil)).

After you created the input files the processing can be started with the following command, where **you only have to set `YOUR_INPUT_FOLDER` and `YOUR_OUTPUT_FOLDER`**:

    docker run --mount type=bind,source=YOUR_INPUT_FOLDER,target=/input --mount type=bind,source=YOUR_OUTPUT_FOLDER,target=/output jenspetersen/hd-glio-auto

The mounts allow the container to access files within. You can also append a number of flags to the command

    -t1sub
    Will create a subtraction map of CT1 - T1 (z-score normalized)

    -v/--verbose
    Turn on verbose output

    -ow/--overwrite
    Allow overwriting of existing files

    -d/--device
    Select GPU (integer number, default=0)

    -np/--no_permissions
    By default the script will copy permissions from the T1 input to the output files. To turn off that behaviours, set this flag.

After a successful run, you should find the following files in your output directory

    segmentation.nii.gz        (The segmentation)
    volumes.txt                (Text file with tumor volumes)
    T1_r2s_bet.nii.gz          (The preprocessed T1 input)
    T1_r2s_bet_mask.nii.gz     (The brain mask for T1)
    CT1_r2s_bet_regT1.nii.gz   (The preprocessed CT1 input)
    T2_r2s_bet_regT1.nii.gz    (The preprocessed T2 input)
    FLAIR_r2s_bet_regT1.nii.gz (The preprocessed FLAIR input)
    T1sub_r2s_bet.nii.gz       (If -t1sub was specified)

## Building the image

If you need to build the image yourself, follow these steps:

1. Clone the repository.
2. Run `docker build` on the repository directory.
                  

