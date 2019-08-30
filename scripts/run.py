#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""

:AUTHOR: Jens Petersen
:ORGANIZATION: Heidelberg University Hospital; German Cancer Research Center
:CONTACT: jens.petersen@dkfz.de
:VERSION: 0.1

"""
# =============================================================================
# IMPORT STATEMENTS
# =============================================================================

import argparse
from datetime import datetime
import os
import re
import shutil
import subprocess as subp
import time
import traceback
import multiprocessing as mp
from functools import partial
import nibabel as nib
import numpy as np

# =============================================================================
# METHODS & CLASSES
# =============================================================================

def reorient2std(file_, overwrite=False):
    outfile = file_.replace(".nii.gz", "_r2s.nii.gz")
    if not os.path.exists(outfile) or overwrite:
        cmd = ['fsl5.0-fslreorient2std', file_, outfile]
        output = subp.check_output(cmd)
    else:
        output = "{} already exists and overwrite=False.".format(outfile)
    return outfile, output


def register_to_t1(file_, reference_file, overwrite=False):
    outfile = file_.replace(".nii.gz", "_regT1.nii.gz")
    if not os.path.exists(outfile) or overwrite:
        cmd = ['fsl5.0-flirt', '-in', file_, '-ref', reference_file, '-out', outfile, '-dof', '6', "-interp", "spline"]
        output = subp.check_output(cmd)
    else:
        output = "{} already exists and overwrite=False.".format(outfile)
    return outfile, output


def run(t1, ct1, t2, flair, output_dir,
        make_t1sub=False,
        verbose=False,
        overwrite=False,
        necrosis_to_background=True,
        copy_permissions=True):

    # switch to output dir to do work and create temporary working dir
    old_wd = os.getcwd()
    os.makedirs(output_dir, exist_ok=True)
    os.chdir(output_dir)
    existing_files = os.listdir(output_dir)
    files_to_remove = [
        "T1.nii",
        "T1.nii.gz",
        "T1_r2s.nii.gz",
        "T1_info.txt",
        "CT1.nii",
        "CT1.nii.gz",
        "CT1_r2s.nii.gz",
        "CT1_r2s_bet.nii.gz",
        "CT1_r2s_bet_mask.nii.gz",
        "CT1_info.txt",
        "T2.nii",
        "T2.nii.gz",
        "T2_r2s.nii.gz",
        "T2_r2s_bet.nii.gz",
        "T2_r2s_bet_mask.nii.gz",
        "T2_info.txt",
        "FLAIR.nii",
        "FLAIR.nii.gz",
        "FLAIR_r2s.nii.gz",
        "FLAIR_r2s_bet.nii.gz",
        "FLAIR_r2s_bet_mask.nii.gz",
        "FLAIR_info.txt",
        "plans.pkl",
        "segmentation_0000.nii.gz",
        "segmentation_0001.nii.gz",
        "segmentation_0002.nii.gz",
        "segmentation_0003.nii.gz",
        "T1_r2s_bet_norm.nii.gz",
        "CT1_r2s_bet_regT1_norm.nii.gz"
    ]
    files_to_keep = [
        "T1_r2s_bet.nii.gz",
        "T1_r2s_bet_mask.nii.gz",
        "CT1_r2s_bet_regT1.nii.gz",
        "T2_r2s_bet_regT1.nii.gz",
        "FLAIR_r2s_bet_regT1.nii.gz",
        "segmentation.nii.gz",
        "volumes.txt"
    ]
    if make_t1sub:
        files_to_keep.append("T1sub_r2s_bet.nii.gz")

    files = [t1, ct1, t2, flair]
    names = ["T1", "CT1", "T2", "FLAIR"]
    filenames_after_preprocessing = ["T1_r2s_bet.nii.gz", "CT1_r2s_bet_regT1.nii.gz", "T2_r2s_bet_regT1.nii.gz", "FLAIR_r2s_bet_regT1.nii.gz"]

    # check if all preprocessed files are already there, if so, skip all of it
    need_preprocess = False
    for f in filenames_after_preprocessing:
        if f not in existing_files:
            need_preprocess = True
    if overwrite:
        need_preprocess = True

    if not need_preprocess:
        files = [os.path.join(output_dir, f) for f in filenames_after_preprocessing]
        if verbose:
            print("All preprocessed inputs for segmentation are available, skipping preprocessing.")
    else:

        # convert to NIfTI if necessary, otherwise copy to output_dir
        for f, file_ in enumerate(files):
            if not overwrite:
                if names[f] + ".nii" in existing_files:
                    files[f] = os.path.join(output_dir, names[f] + ".nii")
                    if verbose:
                        print(names[f] + ".nii already exists, continuing.")
                    continue
                elif names[f] + ".nii.gz" in existing_files:
                    files[f] = os.path.join(output_dir, names[f] + ".nii.gz")
                    if verbose:
                        print(names[f] + ".nii.gz already exists, continuing.")
                    continue
            if os.path.isdir(file_):
                cmd = ["mcverter", "-o", output_dir, "-f", "nifti", "-v", "-n", "-F", names[f], file_]
                output = subp.check_output(cmd)
                if verbose:
                    print("Converted {} to NIfTI with the following output:".format(os.path.basename(file_)))
                    print(output)
                files[f] = os.path.join(output_dir, names[f] + ".nii")
            else:
                new_file = os.path.join(output_dir, os.path.basename(file_))
                shutil.copy(file_, new_file)
                files[f] = new_file
                if verbose:
                    print("Copied {} to output directory.".format(os.path.basename(file_)))

        # zip if necessary
        for f, file_ in enumerate(files):
            if not file_.endswith(".gz"):
                if not overwrite and os.path.basename(file_.replace(".nii", ".nii.gz")) in existing_files:
                    files[f] = file_.replace(".nii", ".nii.gz")
                    if verbose:
                        print(os.path.basename(file_.replace(".nii", ".nii.gz")) + " already exists, not zipping .nii file.")
                    continue
                output = subp.check_output(["gzip", file_])
                if verbose:
                    print("Zipped {} with the following output:".format(os.path.basename(file_)))
                    print(output)
                files[f] = file_.replace(".nii", ".nii.gz")

        # Reorient to standard
        p = mp.Pool(min(len(files), mp.cpu_count()))
        results = p.map(partial(reorient2std, overwrite=overwrite), files)
        files, outputs = list(zip(*results))
        files = list(files)
        if verbose:
            print("Reoriented all files to standard orientation with the following outputs:")
            for output in outputs:
                print(output)

        # Brain extraction (do not parallelize because we run on gpu)
        mask_files = []
        for f, file_ in enumerate(files):
            mask_file = file_.replace(".nii.gz", "_bet_mask.nii.gz")
            mask_files.append(mask_file)
            new_file = file_.replace(".nii.gz", "_bet.nii.gz")
            if not overwrite and os.path.basename(new_file) in existing_files and os.path.basename(mask_file) in existing_files:
                files[f] = new_file
                print("{} and {} already exist, continuing.".format(os.path.basename(new_file), os.path.basename(mask_file)))
                continue
            output1 = subp.check_output(["hd-bet", "-i", file_, "-device", "0"])
            cmd = ["fsl5.0-fslmaths", new_file, "-mas", mask_file, new_file]
            output2 = subp.check_output(cmd)
            files[f] = new_file
            if verbose:
                print("Applied brain extraction for {} with the following output:".format(os.path.basename(file_)))
                print(output)

        # Register to T1
        results = p.map(partial(register_to_t1, reference_file=files[0], overwrite=overwrite), files[1:])
        new_files, outputs = list(zip(*results))
        files[1:] = new_files
        if verbose:
            print("Registered all sequences to T1 with the following outputs:")
            for output in outputs:
                print(output)

    # T1 subtraction map
    if make_t1sub and (overwrite or "T1sub_r2s_bet.nii.gz" not in existing_files):
        t1_file = files[0]
        ct1_file = files[1]
        sub_file = files[0].replace("T1", "T1sub")
        for f in (t1_file, ct1_file):
            nifti = nib.load(f)
            data = nifti.get_data()
            data = data - np.mean(data)
            data = data / np.std(data)
            new = nib.Nifti1Image(data, nifti.affine, nifti.header)
            outname = f.replace(".nii.gz", "_norm.nii.gz")
            nib.save(new, outname)
        cmd = ['fsl5.0-fslmaths',
               ct1_file.replace(".nii.gz", "_norm.nii.gz"),
               '-sub',
               t1_file.replace(".nii.gz", "_norm.nii.gz"),
               sub_file]
        output = subp.check_output(cmd)
        if verbose:
            print("Made T1 subtraction map with output:")
            print(output)


    # prepare Decathlon format for nnU-Net and run segmentation
    if overwrite or "segmentation.nii.gz" not in existing_files:
        for c in range(4):
            new_name = os.path.join(os.path.dirname(files[c]), "segmentation_{:04d}.nii.gz".format(c))
            shutil.copy(files[c], new_name)
        os.chdir("/scripts/segment/nnunet_code/nnUNet/nnunet")
        output = subp.check_output(["python3", "inference/predict_simple.py", "-i", output_dir, "-o", output_dir, "-t", "Task12_BrainTumorIntern"])
        if necrosis_to_background:
            seg_file = os.path.join(output_dir, "segmentation.nii.gz")
            img = nib.load(seg_file)
            data = img.get_data()
            data[data == 3] = 0  # necrosis to background
            nib.save(img, seg_file)
        for c in range(4):
            new_name = os.path.join(os.path.dirname(files[c]), "segmentation_{:04d}.nii.gz".format(c))
            shutil.copy(new_name, files[c])
        os.chdir(output_dir)
        if verbose:
            print("Created segmentation.nii.gz with the following output:")
            print(output)
    elif verbose:
        print("segmentation.nii.gz already exists, continuing.")

    # get volumes
    if overwrite or "volumes.txt" not in existing_files:
        seg_file = os.path.join(output_dir, "segmentation.nii.gz")
        img = nib.load(seg_file)
        data = img.get_data()
        vol_edema = np.sum(data == 1)
        vol_enhancing = np.sum(data == 2)
        with open(os.path.join(output_dir, "volumes.txt"), "w") as outfile:
            outfile.write("volume_non_enhancing_T2_FLAIR_signal_abnormality_mm3: {:.2f}".format(vol_edema))
            outfile.write("\n")
            outfile.write("volume_contrast_enhancing_tumor_mm3: {:.2f}".format(vol_enhancing))
        if verbose:
            print("Created volumes.txt")
    else:
        if verbose:
            print("volumes.txt already exists, continuing.")

    # Remove all files we created and don't want to keep
    for file_ in files_to_remove:
        if file_ in existing_files:
            continue
        else:
            try:
                os.remove(os.path.join(output_dir, file_))
                if verbose:
                    print("Removed {}".format(file_))
            except FileNotFoundError:
                continue

    # Set permissions of files we created
    if copy_permissions:
        for file_ in files_to_keep:
            if file_ in existing_files:
                continue
            else:
                shutil.copymode(t1, os.path.join(output_dir, file_))
                shutil.chown(os.path.join(output_dir, file_), user=os.stat(t1).st_uid, group=os.stat(t1).st_gid)
                if verbose:
                    print("Set permissions for {}".format(file_))

    os.chdir(old_wd)


# =============================================================================
# MAIN METHOD
# =============================================================================

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Glioblastoma Segmentation")
    parser.add_argument("-i", "--input_dir", type=str, required=True, help="Input directory")
    parser.add_argument("-o", "--output_dir", type=str, default=None, help="Output directory")
    parser.add_argument("-t1sub", action="store_true", help="Create T1 subtraction map")
    parser.add_argument("-v", "--verbose", action="store_true", help="Toggle verbose output")
    parser.add_argument("-ow", "--overwrite", action="store_true", help="Overwrite existing files")
    parser.add_argument("-d", "--device", type=str, default="0", help="Select GPU (integer, default=0)")
    # parser.add_argument("-kn", "--keep_necrosis", action="store_true", help="Do not remove necrosis from segmentation")
    parser.add_argument("-np", "--no_permissions", action="store_true", help="Do not adjust permissions of created files (meaning they will be owned by root)")
    args = parser.parse_args()

    if args.output_dir is None:
        output_dir = args.input_dir
    else:
        output_dir = args.output_dir

    inputs = os.listdir(args.input_dir)
    inputs_resolved = []
    for identifier in ("T1", "CT1", "T2", "FLAIR"):
        identifier_options = (identifier + ".nii.gz", identifier + ".nii", identifier)
        for option in identifier_options:
            if option in inputs:
                inputs_resolved.append(option)
                break
        else:
            raise IOError("Could not find input for {}".format(identifier))
    for i in range(4):
        inputs_resolved[i] = os.path.join(args.input_dir, inputs_resolved[i])

    os.environ["CUDA_VISIBLE_DEVICES"] = args.device

    run(*inputs_resolved, output_dir, args.t1sub, args.verbose, args.overwrite, False, not args.no_permissions)

# -*- coding: utf-8 -*-
