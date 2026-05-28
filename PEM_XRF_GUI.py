
# ██████████████████████████████████████████████████████████████████████
# █  GC-MS calculator for ABE mixtures reaction                        █
# █                                                                    █
# █  Debuged with: OpenAI ChatGPT (GPT-5.3)                            █
# █  05.2026                                                           █
# ██████████████████████████████████████████████████████████████████████


# -----------------------------
# Libraries
# -----------------------------
import matplotlib
matplotlib.use('Agg')
from PyQt6 import QtCore, QtGui, QtWidgets,QtSvg
from PyQt6.QtWidgets import QApplication, QDialog, QFileDialog, QMessageBox,QLineEdit
from PyQt6.QtGui import QDoubleValidator
import os
import shutil
import matplotlib.pyplot as plt
import tifffile
import re
from matplotlib.patches import Rectangle
from matplotlib.ticker import FuncFormatter
import numpy as np
import skimage as ski
from skimage import filters
from skimage.filters import threshold_multiotsu
from skimage import io
import pandas as pd
import tempfile
from collections import OrderedDict
import csv



#Font parameters
title_font = {'family': 'arial', 'size': 12}
label_font = {'family': 'arial', 'size': 12}
label_tick = 10
legend_size = 10
rcParams = 'Arial'
dpi_num = 300


def start_project(project_path, sample_name):
    def handle_directory(path):
        """Check and handle directory existence with optional user confirmation."""
        if os.path.exists(path):
            reply = QMessageBox.question(
                None, "Confirm Delete",
                f"The folder '{path}' already exists. Delete it?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.No
            )
            if reply == QMessageBox.StandardButton.No:
                print(f"Using existing directory: {path}")
                return False  # Use existing directory
            shutil.rmtree(path)
            print(f"Deleted and recreated directory: {path}")
        os.makedirs(path, exist_ok=True)
        return True  # Indicates the directory was recreated

    # Main project directory
    dir_before = project_path + "/" + sample_name + "/"
    if not handle_directory(dir_before):
        address_final_plots = dir_before + "Final_Plots/"
        address_plot_of_stacks = dir_before + "Plot_of_stacks/"
        return address_plot_of_stacks,address_final_plots

    # Subdirectories
    address_final_plots = dir_before + "Final_Plots/"
    address_plot_of_stacks = dir_before + "Plot_of_stacks/"

    handle_directory(address_final_plots)
    handle_directory(address_plot_of_stacks)

    return address_plot_of_stacks, address_final_plots


def import_file(address_stack_tiff1,address_final_plots, element, detector,figure_parameters):

    # Plot directory
    path_pic = f"{address_final_plots}Final_Plots_paper_{element}_det_{detector}/"

    if os.path.exists(path_pic):
        shutil.rmtree(path_pic)
        os.mkdir(path_pic)
    else:
        os.mkdir(path_pic)

    # load the real processed image
    uri = address_stack_tiff1

    im = io.imread(uri)  # im[0]...im[12] --> Pt = 0, Ir = 1... Abs = 12

    gray_values_pt = im[0]
    gray_values_ir = im[1]

    # Plots
    plt.rcParams['font.family'] = rcParams

    fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(figure_parameters[0], figure_parameters[1]), facecolor='white')

    ax1 = plt.subplot(121)

    ax1.set_title('Pt emission signal', fontdict=title_font)
    ax1.set_xlabel('X (µm)', fontdict=label_font)
    ax1.set_ylabel('Y (µm)', fontdict=label_font)
    ax1.tick_params(axis='both', labelsize=label_tick)
    p1 = ax1.imshow(np.rot90(gray_values_pt, k=2))
    fig.colorbar(p1, ax=ax1,shrink=figure_parameters[2])
    # Set labels with a specific font style

    ax2 = plt.subplot(122)
    ax2.set_title('Ir emission signal', fontdict=title_font)
    ax2.set_xlabel("X (µm)", fontdict=label_font)
    ax2.set_ylabel("Y(µm)", fontdict=label_font)
    ax2.tick_params(axis='both', labelsize=label_tick)
    p2 = ax2.imshow(np.rot90(gray_values_ir, k=2))
    fig.colorbar(p2, ax=ax2,shrink=figure_parameters[2])

    plt.tight_layout()
    #==========================================
    # Customize the file path
    file_path = path_pic + 'Pt and Ir slices.png'
    # Save only the first subplot
    plt.savefig(file_path, dpi=dpi_num, bbox_inches='tight', pad_inches=0, facecolor=fig.get_facecolor())
    # plt.close()

    # Save to a temporary file as SVG
    temp_file = tempfile.NamedTemporaryFile(suffix=".svg", delete=False)
    plt.savefig(temp_file.name, format="svg", bbox_inches='tight', pad_inches=0)
    plt.close()

    return temp_file.name, gray_values_pt, gray_values_ir  # Return the path to the saved image and gray_values_pt or ir


def import_stack_names(address_stack_tiff1):
    # Load the TIFF stack using tifffile
    uri = address_stack_tiff1

    with tifffile.TiffFile(uri) as tif:
        # Read the TIFF stack into a numpy array
        im_2 = tif.asarray()

        # Collect the labels (titles from metadata)
        image_filenames = []
        for page in tif.pages:
            # Retrieve the metadata for each page (slice)
            metadata = page.tags

            # Look for the 'IJMetadata' tag (ID 50839)
            ij_metadata_tag = metadata.get(50839)
            if ij_metadata_tag is not None:
                # Extract the 'Labels' from the 'IJMetadata'
                labels = ij_metadata_tag.value.get('Labels', [])
                image_filenames.extend(labels)  # Add labels to the list

    #print(f"Image filenames: {image_filenames}")

    # If no filenames found, generate default names for the slices
    if not image_filenames:
        image_filenames = [f"slice_{i}" for i in range(len(im_2))]

    return image_filenames  # Return image_filenames


def thresholding_blur(address_plot_of_stacks,element, gray_values_pt, gray_values_ir, thresh_method, val_pt_manual,
                      val_ir_manual):
    # Directory
    path_1 = address_plot_of_stacks + "Plot_of_stacks_" + element + "/"

    if os.path.exists(path_1):
        shutil.rmtree(path_1)
        os.mkdir(path_1)
    else:
        os.mkdir(path_1)

    # convert the image to dataframe
    gray_values_pt_DF = pd.DataFrame(gray_values_pt)
    gray_values_ir_DF = pd.DataFrame(gray_values_ir)

    # blur the image to denoise
    blurred_gray_pt = ski.filters.gaussian(gray_values_pt, sigma=0.8)
    blurred_gray_ir = ski.filters.gaussian(gray_values_ir, sigma=0.8)

    if thresh_method == "Manual":
        # Use manual threshold values
        val_pt = val_pt_manual
        val_ir = val_ir_manual

    elif thresh_method == "Minimum":
        # Use minimum thresholding
        val_pt = filters.threshold_minimum(blurred_gray_pt)
        val_ir = filters.threshold_minimum(blurred_gray_ir)
    elif thresh_method == "Triangle":
        # Use minimum thresholding
        val_pt = filters.threshold_triangle(blurred_gray_pt)
        val_ir = filters.threshold_triangle(blurred_gray_ir)
    elif thresh_method == "Otsu":
        # Use minimum thresholding
        val_pt = filters.threshold_otsu(blurred_gray_pt)
        val_ir = filters.threshold_otsu(blurred_gray_ir)
    else:
        raise ValueError("Invalid thresholding method. Choose 'manual' or 'minimum'.")

    hist_pt, bins_center_pt = np.histogram(gray_values_pt, bins=65536)
    hist_ir, bins_center_ir = np.histogram(gray_values_ir, bins=65536)

    # Adjust bin centers to match histogram counts
    bins_center_pt = bins_center_pt[:-1]  # Drop the last edge
    bins_center_ir = bins_center_ir[:-1]

    ## Plots
    plt.rcParams['font.family'] = rcParams

    fig, ax = plt.subplots(nrows=2, ncols=2, figsize=(10, 8), facecolor='white')

    ax1 = plt.subplot(2, 2, 1)
    ax1.set_title('Blurred grays pt', fontdict=title_font)
    ax1.set_xlabel('X (µm)', fontdict=label_font)
    ax1.set_ylabel('Y (µm)', fontdict=label_font)
    ax1.tick_params(axis='both', labelsize=label_tick)
    plt.imshow(blurred_gray_pt, cmap='gray')

    ax2 = plt.subplot(2, 2, 2)
    ax2.set_title('Binary mask pt', fontdict=title_font)
    ax2.set_xlabel('X (µm)', fontdict=label_font)
    ax2.set_ylabel('Y (µm)', fontdict=label_font)
    ax2.tick_params(axis='both', labelsize=label_tick)
    plt.imshow(blurred_gray_pt < val_pt, cmap='gray')

    ax3 = plt.subplot(2, 2, 3)
    ax3.set_title('Blurred grays ir')
    ax3.set_xlabel('X (µm)', fontdict=label_font)
    ax3.set_ylabel('Y (µm)', fontdict=label_font)
    ax3.tick_params(axis='both', labelsize=label_tick)
    plt.imshow(blurred_gray_ir, cmap='gray')

    ax4 = plt.subplot(2, 2, 4)
    ax4.set_title('Binary mask ir', fontdict=title_font)
    ax4.set_xlabel('X (µm)', fontdict=label_font)
    ax4.set_ylabel('Y (µm)', fontdict=label_font)
    ax4.tick_params(axis='both', labelsize=label_tick)
    plt.imshow(blurred_gray_ir < val_ir, cmap='gray')

    plt.tight_layout()

    # Save to a temporary file as SVG
    temp_file_2 = tempfile.NamedTemporaryFile(suffix=".svg", delete=False)
    plt.savefig(temp_file_2.name, format="svg", bbox_inches='tight', pad_inches=0)
    plt.close()

    fig_2, ax = plt.subplots(nrows=1, ncols=1, figsize=(5, 4), facecolor='white')
    ax1 = plt.subplot(1, 1, 1)
    ax1.set_title('Histogram of gray values for pt')
    ax1.set_xlim(0, 1000)
    ax1.set_ylim(0, 100)
    ax1.set_xlabel("Gray scale values (a.u.)", fontdict=label_font)
    ax1.set_ylabel("Frequency (a.u.)", fontdict=label_font)
    ax1.tick_params(axis='both', labelsize=label_tick)

    clipped_bins_center_pt = np.clip(bins_center_pt, 0, 1000)
    clipped_hist_pt = np.clip(hist_pt,0, 100)
    plt.plot(clipped_bins_center_pt , clipped_hist_pt, lw=2)
    plt.axvline(val_pt, color='k', ls='--')
    plt.tight_layout()

    # Save to a temporary file as SVG
    temp_file_3 = tempfile.NamedTemporaryFile(suffix=".svg", delete=False)
    plt.savefig(temp_file_3.name, format="svg", bbox_inches='tight', pad_inches=0)
    plt.close()

    fig_3, ax = plt.subplots(nrows=1, ncols=1, figsize=(5, 4), facecolor='white')
    ax1 = plt.subplot(1, 1, 1)
    ax1.title.set_text('Histogram of gray values for ir')
    ax1.set_xlim(0, 1000)
    ax1.set_ylim(0, 100)
    ax1.set_xlabel("Gray scale values (a.u.)", fontdict=label_font)
    ax1.set_ylabel("Frequency (a.u.)", fontdict=label_font)
    ax1.tick_params(axis='both', labelsize=label_tick)

    clipped_bins_center_ir = np.clip(bins_center_ir, 0, 1000)
    clipped_hist_ir = np.clip(hist_ir,0, 100)
    plt.plot(clipped_bins_center_ir , clipped_hist_ir, lw=2)
    plt.axvline(val_ir, color='k', ls='--')
    plt.tight_layout()

    # Save to a temporary file as SVG
    temp_file_4 = tempfile.NamedTemporaryFile(suffix=".svg", delete=False)
    plt.savefig(temp_file_4.name, format="svg", bbox_inches='tight', pad_inches=0)
    plt.close()

    return temp_file_2.name, temp_file_3.name, temp_file_4.name, val_pt, val_ir, blurred_gray_pt, blurred_gray_ir  # Return the path to the saved image and other variables

def binary_plots(address_final_plots, element, detector, val_pt, val_ir, blurred_gray_pt, blurred_gray_ir,figure_parameters_blur):

    # Plot directory
    path_pic = f"{address_final_plots}Final_Plots_paper_{element}_det_{detector}/"

    # Binary mask -----------------------------------------------------------------------
    t_pt = val_pt
    t_ir = val_ir

    binary_mask_nafion_pt = blurred_gray_pt < t_pt
    binary_mask_nafion_ir = blurred_gray_ir < t_ir

    # Split cat layers
    def split_list(a_list):
        half = len(a_list) // 2
        return a_list[:half], a_list[half:]

    binary_mask_nafion_pt_half_top, binary_mask_nafion_pt_half_bottom = split_list(binary_mask_nafion_pt)
    binary_mask_nafion_pt_half_top_DF = pd.DataFrame(binary_mask_nafion_pt_half_top)

    binary_mask_nafion_ir_half_top, binary_mask_nafion_ir_half_bottom = split_list(binary_mask_nafion_ir)
    binary_mask_nafion_ir_half_bottom_DF = pd.DataFrame(binary_mask_nafion_ir_half_bottom)

    #### Mixing both coordinates from different slices from the stack
    binary_mask_nafion = pd.concat([binary_mask_nafion_pt_half_top_DF, binary_mask_nafion_ir_half_bottom_DF], axis=0)
    binary_mask_cat = ~binary_mask_nafion

    # Plot2-------------------------------------------------------------------------------

    plt.rcParams['font.family'] = rcParams

    plt.figure(figsize=(figure_parameters_blur[0], figure_parameters_blur[1]))

    ax1 = plt.subplot(121)
    ax1.set_title('Binary mask nafion', fontdict=title_font)
    ax1.set_xlabel('X (µm)', fontdict=label_font)
    ax1.set_ylabel('Y (µm)', fontdict=label_font)
    ax1.tick_params(axis='both', labelsize=label_tick)
    legend_labels = ['Excess', 'Catalysts']
    legend_elements = [Rectangle((0, 0), 1, 1, color='yellow', lw=2),
                       Rectangle((0, 0), 1, 1, color='purple', lw=2)]
    ax1.legend(legend_elements, legend_labels, loc='upper right', fontsize=legend_size)
    plt.imshow(np.rot90(binary_mask_nafion, k=2))

    # ------------------------------------------------------------------------------------

    ax2 = plt.subplot(122)
    ax2.set_title('Binary mask cat', fontdict=title_font)
    ax2.set_xlabel('X (µm)', fontdict=label_font)
    ax2.set_ylabel('Y (µm)', fontdict=label_font)
    ax2.tick_params(axis='both', labelsize=label_tick)
    legend_labels = ['Catalysts', 'Excess']
    legend_elements = [Rectangle((0, 0), 1, 1, color='yellow', lw=2),
                       Rectangle((0, 0), 1, 1, color='purple', lw=2)]
    ax2.legend(legend_elements, legend_labels, loc='upper right', fontsize=legend_size)
    plt.imshow(np.rot90(binary_mask_cat, k=2))

    plt.tight_layout()

    ##==========================================
    # Customize the file path
    file_path = path_pic + 'Cat and excess.png'
    plt.savefig(file_path, dpi=dpi_num)

    # Save to a temporary file as SVG
    temp_file_5 = tempfile.NamedTemporaryFile(suffix=".svg", delete=False)
    plt.savefig(temp_file_5.name, format="svg", bbox_inches='tight', pad_inches=0)
    plt.close()

    # Return the file and DataFrames
    binary_mask_nafion_DF = pd.DataFrame(binary_mask_nafion)
    binary_mask_cat_DF = pd.DataFrame(binary_mask_cat)

    return temp_file_5.name, binary_mask_nafion_DF, binary_mask_cat_DF, binary_mask_nafion, binary_mask_cat


def segment(address_stack_tiff2, address_final_plots, address_plot_of_stacks, element, listed_elements, detector, binary_mask_nafion, binary_mask_cat,figure_parameters):

    # Plot directory
    uri2 = address_stack_tiff2
    path_pic = f"{address_final_plots}Final_Plots_paper_{element}_det_{detector}/"
    path_1 = address_plot_of_stacks + "Plot_of_stacks_" + element + "/"

    # Find the position of the element in the list
    if element in listed_elements:
        position = listed_elements.index(element)
    else:
        print(f"{element} is not in the listed_elements.")

    # convert the real image to dataframe
    im = io.imread(uri2)  # im[0]...im[13] --> Pt = 0, Ir = 1... Ba = 12
    real_im = im[position]  # change number based on what you want to analyse (element selection at the beginning)

    # Plots ----------------------------------------------------------------------------------------
    plt.rcParams['font.family'] = rcParams

    # Original sample
    fig, ax = plt.subplots(nrows=1, ncols=4, figsize=(figure_parameters[4], figure_parameters[5]))
    ax1 = plt.subplot(141)
    ax1.set_title('Original sample', fontdict=title_font)
    ax1.set_xlabel('X (µm)', fontdict=label_font)
    ax1.set_ylabel('Y (µm)', fontdict=label_font)
    ax1.tick_params(axis='both', labelsize=label_tick)
    p1 = plt.imshow(np.rot90(real_im, k=2))
    fig.colorbar(p1, ax=ax1, shrink=figure_parameters[0])

    # Nafion Layer
    selection1 = real_im.copy()
    selection1[~binary_mask_nafion] = 0

    ax2 = plt.subplot(142)
    ax2.set_title('Segmented Nafion and Epoxy', fontdict=title_font)
    ax2.set_xlabel('X (µm)', fontdict=label_font)
    ax2.set_ylabel('Y (µm)', fontdict=label_font)
    ax2.tick_params(axis='both', labelsize=label_tick)
    p2 = plt.imshow(np.rot90(selection1, k=2))
    fig.colorbar(p2, ax=ax2, shrink=figure_parameters[1])

    # Catalyst Layers --> it is difficult to visualize both cat layers, but they are --> .csv has gray values
    selection2 = real_im.copy()
    selection2[~binary_mask_cat] = 0

    # Split cat layers
    def split_list(a_list):
        half = len(a_list) // 2
        return a_list[:half], a_list[half:]

    selectionPt, selectionIr = split_list(selection2)

    # First half Pt
    ax3 = plt.subplot(143)
    ax3.set_title('Pt catalyst', fontdict=title_font)
    ax3.set_xlabel('X (µm)', fontdict=label_font)
    ax3.set_ylabel('Y (µm)', fontdict=label_font)
    ax3.tick_params(axis='both', labelsize=label_tick)
    p3 = plt.imshow(np.rot90(selectionPt, k=2))
    fig.colorbar(p3, ax=ax3, shrink=figure_parameters[2])

    # Second half Ir
    ax4 = plt.subplot(144)
    ax4.set_title('Ir catalyst', fontdict=title_font)
    ax4.set_xlabel('X (µm)', fontdict=label_font)
    ax4.set_ylabel('Y (µm)', fontdict=label_font)
    ax4.tick_params(axis='both', labelsize=label_tick)
    p4 = plt.imshow(np.rot90(selectionIr, k=2))
    fig.colorbar(p4, ax=ax4, shrink=figure_parameters[3])

    plt.tight_layout()

    ##==========================================
    # Customize the file path
    file_path = path_pic + 'Sample segmented_epoxy.png'
    plt.savefig(file_path, dpi=300)

    # Save to a temporary file as SVG
    temp_file_6 = tempfile.NamedTemporaryFile(suffix=".svg", delete=False)
    plt.savefig(temp_file_6.name, format="svg", bbox_inches='tight', pad_inches=0)
    plt.close()

    # Saving .csv files -----------------------------------------------------------------------------
    selection1_DF = pd.DataFrame(selection1)  # nafion
    selection1_DF.to_csv(path_1 + "gray_values_nafion_DF.csv")

    selectionPt_DF = pd.DataFrame(selectionPt)  # cat
    selectionPt_DF.to_csv(path_1 + "gray_values_Pt_DF.csv")

    selectionIr_DF = pd.DataFrame(selectionIr)  # cat
    selectionIr_DF.to_csv(path_1 + "gray_values_Ir_DF.csv")

    gray_values_real_DF = pd.DataFrame(real_im)
    gray_values_real_DF.to_csv(path_1 + "gray_values_real_DF.csv")

    return temp_file_6.name, selection1_DF, selectionPt_DF, selectionIr_DF, gray_values_real_DF, selectionPt, selectionIr, real_im


# block of sibling functions: they give the original contour before the fine tune ****
def Pt_catalyst_contour(address_plot_of_stacks, element, selectionPt_DF, selection1_DF):

    ## Collect superior border and last border  and then average to get X and Y coordinates of the average
    path_pt = address_plot_of_stacks + "Plot_of_stacks_" + element + "/Plot_of_stacks_Pt/"

    if os.path.exists(path_pt):
        shutil.rmtree(path_pt)
        os.mkdir(path_pt)
    else:
        os.mkdir(path_pt)

        # define '0' matrix to be transformed in a Data frame full of coordinates of catalyst border values
    border_top_Pt = [[0] * 3] * len(selectionPt_DF.columns)  # border_top_Pt = Border values for the Top Pt
    border_top_Pt_DF = pd.DataFrame(border_top_Pt)
    border_top_Pt_DF = border_top_Pt_DF.rename(columns={0: "X_border", 1: "Y_border", 2: "Intensity_border"})

    # define '0' matrix to be transformed in a Data frame full of coordinates of catalyst border values
    border_bottom_Pt = [[0] * 3] * len(selectionPt_DF.columns)  # border_top_Pt = Border values for the Top Pt
    border_bottom_Pt_DF = pd.DataFrame(border_bottom_Pt)
    border_bottom_Pt_DF = border_bottom_Pt_DF.rename(columns={0: "X_border", 1: "Y_border", 2: "Intensity_border"})

    skip = 0

    for j in range(0, len(selection1_DF.columns) - 1):

        a_p = selectionPt_DF.iloc[:, j]
        a = a_p.to_frame()
        a = a.rename(columns={j: "Intensity"})
        count = np.count_nonzero(a_p)

        if count >= 10:
            a = a.dropna()  # cleans Dataframe 'a_1' with NaN values
            a = a[a['Intensity'] != 0]
            a.to_csv(path_pt + "/Plot_of_stack_Pt_{}.csv".format(j + 1))

            ## User defined variable ##

            fine_tune = 10  # add or subtracts 'x' pixels from y coordinate -- shrinking or expanding the Nafion membrane region

            # First border for Pt
            border_top_Pt_DF['Intensity_border'].values[j + 1] = a['Intensity'].values[0]
            border_top_Pt_DF['Y_border'].values[j + 1] = a.first_valid_index() - fine_tune
            border_top_Pt_DF['X_border'].values[j + 1] = j + 1

            # Second border for Pt
            size_frame = len(a) - 1
            border_bottom_Pt_DF['Intensity_border'].values[j + 1] = a['Intensity'].values[size_frame]
            border_bottom_Pt_DF['Y_border'].values[j + 1] = a.index[-1] - fine_tune
            border_bottom_Pt_DF['X_border'].values[j + 1] = j + 1

        else:
            skip = skip + 1
            a = a.dropna()  # cleans Dataframe 'a_1' with NaN values
            a = a[a['Intensity'] != 0]

    j = j + 1

    # print(border_top_Pt_DF)
    # print(border_bottom_Pt_DF)

    # TOP --------------------------------------------------------------------------------------------
    contour_first_Pt = [0] * (len(border_top_Pt_DF)) * 2
    contour_first_Pt_DF = pd.DataFrame(contour_first_Pt)
    contour_first_Pt_DF = contour_first_Pt_DF.rename(columns={0: "Coordinates_1D"})

    for i in range(0, len(border_top_Pt_DF)):
        contour_first_Pt_DF['Coordinates_1D'].values[i * 2] = border_top_Pt_DF['X_border'].values[i]

    for i in range(1, len(border_top_Pt_DF) + 1):
        contour_first_Pt_DF['Coordinates_1D'].values[2 * i - 1] = border_top_Pt_DF['Y_border'].values[i - 1]

    contour_first_Pt_DF = contour_first_Pt_DF.dropna()  # cleans NaN values
    contour_first_Pt_DF = contour_first_Pt_DF[contour_first_Pt_DF['Coordinates_1D'] != 0]  # cleans 0 values

    list_1c = contour_first_Pt_DF['Coordinates_1D'].tolist()

    # Bottom ------------------------------------------------------------------------------------------
    contour_second_Pt = [0] * (len(border_bottom_Pt_DF)) * 2
    contour_second_Pt_DF = pd.DataFrame(contour_second_Pt)
    contour_second_Pt_DF = contour_second_Pt_DF.rename(columns={0: "Coordinates_1D"})

    for i in range(0, len(border_bottom_Pt_DF)):
        contour_second_Pt_DF['Coordinates_1D'].values[i * 2] = border_bottom_Pt_DF['Y_border'].values[i]

    for i in range(1, len(border_bottom_Pt_DF) + 1):
        contour_second_Pt_DF['Coordinates_1D'].values[2 * i - 1] = border_bottom_Pt_DF['X_border'].values[i - 1]

    contour_second_Pt_DF = contour_second_Pt_DF.dropna()  # cleans NaN values
    contour_second_Pt_DF = contour_second_Pt_DF[contour_second_Pt_DF['Coordinates_1D'] != 0]  # cleans 0 values
    contour_second_Pt_DF = contour_second_Pt_DF.iloc[::-1]  # reverse row's order for better data acquisition

    list_2c = contour_second_Pt_DF['Coordinates_1D'].tolist()

    # Concatenate ------------------------------------------------------------------------------------------
    bigDF_Pt = pd.concat([contour_first_Pt_DF, contour_second_Pt_DF])
    biglist_Pt = bigDF_Pt['Coordinates_1D'].tolist()

    return biglist_Pt


def Ir_catalyst_contour(address_plot_of_stacks, element, selectionIr_DF, selectionPt):

    ## Collect superior border and last border  and then average to get X and Y coordinates of the average
    path_Ir = address_plot_of_stacks + "Plot_of_stacks_" + element + "/Plot_of_stacks_Ir/"

    if os.path.exists(path_Ir):
        shutil.rmtree(path_Ir)
        os.mkdir(path_Ir)
    else:
        os.mkdir(path_Ir)

        # define '0' matrix to be transformed in a Data frame full of coordinates of catalyst border values
    border_top_Ir = [[0] * 3] * len(selectionIr_DF.columns)  # border_top_Ir = Border values for the Top Ir
    border_top_Ir_DF = pd.DataFrame(border_top_Ir)
    border_top_Ir_DF = border_top_Ir_DF.rename(columns={0: "X_border", 1: "Y_border", 2: "Intensity_border"})

    # define '0' matrix to be transformed in a Data frame full of coordinates of catalyst border values
    border_bottom_Ir = [[0] * 3] * len(selectionIr_DF.columns)  # border_top_Ir = Border values for the Top Ir
    border_bottom_Ir_DF = pd.DataFrame(border_bottom_Ir)
    border_bottom_Ir_DF = border_bottom_Ir_DF.rename(columns={0: "X_border", 1: "Y_border", 2: "Intensity_border"})

    skip = 0

    for j in range(0, len(selectionIr_DF.columns) - 1):

        a_p = selectionIr_DF.iloc[:, j]
        a = a_p.to_frame()
        a = a.rename(columns={j: "Intensity"})
        count = np.count_nonzero(a_p)

        if count >= 10:
            a = a.dropna()  # cleans Dataframe 'a_1' with NaN values
            a = a[a['Intensity'] != 0]
            a.to_csv(path_Ir + "/Plot_of_stack_ir_{}.csv".format(j + 1))

            ## User defined variable ##

            fine_tune = 10  # add or subtracts 'x' pixels from y coordinate -- shrinking or expanding the Nafion membrane region

            # First border for Ir
            border_top_Ir_DF['Intensity_border'].values[j + 1] = a['Intensity'].values[0]
            border_top_Ir_DF['Y_border'].values[j + 1] = a.first_valid_index() + len(selectionPt) - fine_tune
            border_top_Ir_DF['X_border'].values[j + 1] = j + 1

            # Second border for Ir
            size_frame = len(a) - 1
            border_bottom_Ir_DF['Intensity_border'].values[j + 1] = a['Intensity'].values[size_frame]
            border_bottom_Ir_DF['Y_border'].values[j + 1] = a.index[-1] + len(selectionPt) - fine_tune
            border_bottom_Ir_DF['X_border'].values[j + 1] = j + 1

        else:
            skip = skip + 1
            a = a.dropna()  # cleans Dataframe 'a_1' with NaN values
            a = a[a['Intensity'] != 0]

    j = j + 1

    # print(border_top_Ir_DF)
    # print(border_bottom_Ir_DF)

    # TOP --------------------------------------------------------------------------------------------
    contour_first_Ir = [0] * (len(border_top_Ir_DF)) * 2
    contour_first_Ir_DF = pd.DataFrame(contour_first_Ir)
    contour_first_Ir_DF = contour_first_Ir_DF.rename(columns={0: "Coordinates_1D"})

    for i in range(0, len(border_top_Ir_DF)):
        contour_first_Ir_DF['Coordinates_1D'].values[i * 2] = border_top_Ir_DF['X_border'].values[i]

    for i in range(1, len(border_top_Ir_DF) + 1):
        contour_first_Ir_DF['Coordinates_1D'].values[2 * i - 1] = border_top_Ir_DF['Y_border'].values[i - 1]

    contour_first_Ir_DF = contour_first_Ir_DF.dropna()  # cleans NaN values
    contour_first_Ir_DF = contour_first_Ir_DF[contour_first_Ir_DF['Coordinates_1D'] != 0]  # cleans 0 values

    list_1c = contour_first_Ir_DF['Coordinates_1D'].tolist()

    # Bottom ------------------------------------------------------------------------------------------
    contour_second_Ir = [0] * (len(border_bottom_Ir_DF)) * 2
    contour_second_Ir_DF = pd.DataFrame(contour_second_Ir)
    contour_second_Ir_DF = contour_second_Ir_DF.rename(columns={0: "Coordinates_1D"})

    for i in range(0, len(border_bottom_Ir_DF)):
        contour_second_Ir_DF['Coordinates_1D'].values[i * 2] = border_bottom_Ir_DF['Y_border'].values[i]

    for i in range(1, len(border_bottom_Ir_DF) + 1):
        contour_second_Ir_DF['Coordinates_1D'].values[2 * i - 1] = border_bottom_Ir_DF['X_border'].values[i - 1]

    contour_second_Ir_DF = contour_second_Ir_DF.dropna()  # cleans NaN values
    contour_second_Ir_DF = contour_second_Ir_DF[contour_second_Ir_DF['Coordinates_1D'] != 0]  # cleans 0 values
    contour_second_Ir_DF = contour_second_Ir_DF.iloc[::-1]  # reverse row's order for better data acquisition

    list_2c = contour_second_Ir_DF['Coordinates_1D'].tolist()

    # Concatenate ------------------------------------------------------------------------------------------
    bigDF_Ir = pd.concat([contour_first_Ir_DF, contour_second_Ir_DF])
    biglist_Ir = bigDF_Ir['Coordinates_1D'].tolist()

    return biglist_Ir, contour_second_Ir


def Nafion_membrane_contour(address_plot_of_stacks, element, selection1_DF, contour_second_Ir):

    ## Collect superior border and last border  and then average to get X and Y coordinates of the average
    path_nafion = address_plot_of_stacks + "Plot_of_stacks_" + element + "/Plot_of_stacks_Nafion/"

    if os.path.exists(path_nafion):
        shutil.rmtree(path_nafion)
        os.mkdir(path_nafion)
    else:
        os.mkdir(path_nafion)

        # define '0' matrix to be transformed in a Data frame full of coordinates of catalyst border values
    border_top_nafion = [[0] * 3] * len(selection1_DF.columns)  # border_top_nafion = Border values for the Top Ir
    border_top_nafion_DF = pd.DataFrame(border_top_nafion)
    border_top_nafion_DF = border_top_nafion_DF.rename(columns={0: "X_border", 1: "Y_border", 2: "Intensity_border"})

    # define '0' matrix to be transformed in a Data frame full of coordinates of catalyst border values
    border_bottom_nafion = [[0] * 3] * len(
        selection1_DF.columns)  # border_bottom_nafion = Border values for the bottom Pt
    border_bottom_nafion_DF = pd.DataFrame(border_bottom_nafion)
    border_bottom_nafion_DF = border_bottom_nafion_DF.rename(
        columns={0: "X_border", 1: "Y_border", 2: "Intensity_border"})

    # generate dataframe only for membrane layer
    nafion_membrane = [[0] * len(selection1_DF.columns)] * len(selection1_DF)
    nafion_membrane_DF = pd.DataFrame(nafion_membrane)

    skip = 0

    for j in range(0, len(selection1_DF.columns) - 1):
        a_p = selection1_DF.iloc[:, j]
        a = a_p.to_frame()
        a = a.rename(columns={j: "Intensity"})
        count = np.count_nonzero(a_p)

        # First index where values is c
        idx = a.index[a['Intensity'].eq(0)].min()
        idx2 = a.index[a['Intensity'].eq(0)].max()

        a.loc[a.index < idx] = 0  # zero before top Pt border
        a.loc[a.index > idx2] = 0  # zero after bottom Ir border

        nafion_membrane_DF[j] = a

        if count >= 10:

            a = a[idx:idx2]  # Drop first borders
            a = a.dropna()  # cleans Dataframe 'a_1' with NaN values
            a = a[a['Intensity'] != 0]
            a.to_csv(path_nafion + "/Plot_of_stack_nafion_{}.csv".format(j + 1))

            ## User defined variable ##

            fine_tune = 0  # add or subtracts 'x' pixels from y coordinate -- shrinking or expanding the Nafion membrane region

            # First border for Ir
            border_top_nafion_DF['Intensity_border'].values[j + 1] = a['Intensity'].values[0]
            border_top_nafion_DF['Y_border'].values[j + 1] = a.first_valid_index() - fine_tune
            border_top_nafion_DF['X_border'].values[j + 1] = j + 1

            # Second border for Ir
            size_frame = len(a) - 1
            border_bottom_nafion_DF['Intensity_border'].values[j + 1] = a['Intensity'].values[size_frame]
            border_bottom_nafion_DF['Y_border'].values[j + 1] = a.index[-1] - fine_tune
            border_bottom_nafion_DF['X_border'].values[j + 1] = j + 1

        else:
            skip = skip + 1
            a = a.dropna()  # cleans Dataframe 'a_1' with NaN values
            a = a[a['Intensity'] != 0]

    j = j + 1

    # TOP --------------------------------------------------------------------------------------------
    contour_first_nafion = [0] * (len(border_top_nafion_DF)) * 2
    contour_first_nafion_DF = pd.DataFrame(contour_first_nafion)
    contour_first_nafion_DF = contour_first_nafion_DF.rename(columns={0: "Coordinates_1D"})

    for i in range(0, len(border_top_nafion_DF)):
        contour_first_nafion_DF['Coordinates_1D'].values[i * 2] = border_top_nafion_DF['X_border'].values[i]

    for i in range(1, len(border_top_nafion_DF) + 1):
        contour_first_nafion_DF['Coordinates_1D'].values[2 * i - 1] = border_top_nafion_DF['Y_border'].values[i - 1]

    contour_first_nafion_DF = contour_first_nafion_DF.dropna()  # cleans NaN values
    contour_first_nafion_DF = contour_first_nafion_DF[contour_first_nafion_DF['Coordinates_1D'] != 0]  # cleans 0 values

    list_1c = contour_first_nafion_DF['Coordinates_1D'].tolist()

    # Bottom ------------------------------------------------------------------------------------------
    contour_second_nafion = [0] * (len(border_bottom_nafion_DF)) * 2
    contour_second_nafion_DF = pd.DataFrame(contour_second_Ir)
    contour_second_nafion_DF = contour_second_nafion_DF.rename(columns={0: "Coordinates_1D"})

    for i in range(0, len(border_bottom_nafion_DF)):
        contour_second_nafion_DF['Coordinates_1D'].values[i * 2] = border_bottom_nafion_DF['Y_border'].values[i]

    for i in range(1, len(border_bottom_nafion_DF) + 1):
        contour_second_nafion_DF['Coordinates_1D'].values[2 * i - 1] = border_bottom_nafion_DF['X_border'].values[i - 1]

    contour_second_nafion_DF = contour_second_nafion_DF.dropna()  # cleans NaN values
    contour_second_nafion_DF = contour_second_nafion_DF[
        contour_second_nafion_DF['Coordinates_1D'] != 0]  # cleans 0 values
    contour_second_nafion_DF = contour_second_nafion_DF.iloc[::-1]  # reverse row's order for better data acquisition

    list_2c = contour_second_nafion_DF['Coordinates_1D'].tolist()

    # Concatenate ------------------------------------------------------------------------------------------
    bigDF_nafion = pd.concat([contour_first_nafion_DF, contour_second_nafion_DF])
    biglist_nafion = bigDF_nafion['Coordinates_1D'].tolist()

    return biglist_nafion, nafion_membrane_DF
# ***

def fine_tune(address_final_plots, element, detector, real_im, nafion_membrane_DF, selectionPt, selectionIr, fine_tune_var,figure_parameters_fine_tune):

    # Plot directory
    path_pic = f"{address_final_plots}Final_Plots_paper_{element}_det_{detector}/"

    # Plots ----------------------------------------------------------------------------------------
    plt.rcParams['font.family'] = rcParams
    fig = plt.figure()
    fig, ax = plt.subplots(nrows=1, ncols=4, figsize=(figure_parameters_fine_tune[4], figure_parameters_fine_tune[5]))
    ax1 = plt.subplot(141)
    ax1.title.set_text('Original sample')
    ax1.set_xlabel("X (µm)")
    ax1.set_ylabel("Y(µm)")
    ax1.tick_params(axis='both', labelsize=label_tick)
    p1 = plt.imshow(np.rot90(real_im, k=2))
    fig.colorbar(p1, ax=ax1, shrink=figure_parameters_fine_tune[0])

    # generates dataframe only for membrane layer and remove some pixels due to the high intensity
    nafion_membrane1 = [[0] * len(nafion_membrane_DF.columns)] * len(nafion_membrane_DF)
    nafion_membrane_DF1 = pd.DataFrame(nafion_membrane1)

    for j in range(0, len(nafion_membrane_DF.columns) - 1):
        a_p1 = nafion_membrane_DF.iloc[:, j]
        a1 = a_p1.to_frame()
        a1 = a1.rename(columns={j: "Intensity"})
        count = np.count_nonzero(a_p1)

        # First index where values is c
        Fine_tune_FIN = fine_tune_var
        idx1 = a1.index[a1['Intensity'] != 0].min() + Fine_tune_FIN
        idx3 = a1.index[a1['Intensity'] != 0].max() - Fine_tune_FIN

        a1.loc[a1.index < idx1] = 0  # zero before top Pt border
        a1.loc[a1.index > idx3] = 0  # zero after bottom Ir border

        nafion_membrane_DF1[j] = a1
    j = j + 1

    # Nafion Layer
    ax2 = plt.subplot(142)
    ax2.title.set_text('Segmented Nafion')
    ax2.set_xlabel("X (µm)")
    ax2.set_ylabel("Y(µm)")
    ax2.tick_params(axis='both', labelsize=label_tick)
    p2 = plt.imshow(np.rot90(nafion_membrane_DF1, k=2))
    fig.colorbar(p2, ax=ax2, shrink=figure_parameters_fine_tune[1])

    # First half Pt
    ax3 = plt.subplot(143)
    ax3.title.set_text('Pt catalyst')
    ax3.set_xlabel("X (µm)")
    ax3.set_ylabel("Y(µm)")
    ax3.tick_params(axis='both', labelsize=label_tick)
    p3 = plt.imshow(np.rot90(selectionPt, k=2))
    fig.colorbar(p3, ax=ax3, shrink=figure_parameters_fine_tune[2])

    # Second half Ir
    ax4 = plt.subplot(144)
    ax4.title.set_text('Ir catalyst')
    ax4.set_xlabel("X (µm)")
    ax4.set_ylabel("Y(µm)")
    ax4.tick_params(axis='both', labelsize=label_tick)
    p4 = plt.imshow(np.rot90(selectionIr, k=2))
    fig.colorbar(p4, ax=ax4, shrink=figure_parameters_fine_tune[3])

    plt.tight_layout()

    # ==========================================
    # Customize the file path
    file_path = path_pic + 'Sample segmented.png'
    plt.savefig(file_path, dpi=dpi_num)

    # Save to a temporary file as SVG
    temp_file_7 = tempfile.NamedTemporaryFile(suffix=".svg", delete=False)
    plt.savefig(temp_file_7.name, format="svg", bbox_inches='tight', pad_inches=0)
    plt.close()

    return temp_file_7.name, nafion_membrane_DF1


def filter_dust(nafion_membrane_DF1, filter_dust_value, plot_3d_var, plot_multi_otsu_var_1, plot_hist_otsu_var_1,legend_loc):

    # Nafion Layer
    nafion_membrane_DF2 = nafion_membrane_DF1.copy()
    binary_mask2_nafion = nafion_membrane_DF2 < filter_dust_value
    nafion_membrane_DF2[~binary_mask2_nafion] = 0
    # Plots -------------------------------------------------
    plt.rcParams['font.family'] = rcParams

    #membrane plot-----------------
    fig, ax = plt.subplots(nrows=1, ncols=1,figsize=(plot_multi_otsu_var_1[3], plot_multi_otsu_var_1[4]))

    ax.set_title('Nafion and epoxy', fontdict=title_font)
    ax.set_xlabel('X (µm)', fontdict=label_font)
    ax.set_ylabel('Y (µm)', fontdict=label_font)
    ax.imshow(np.rot90(nafion_membrane_DF2, k=2))

    # Save to a temporary file as SVG
    temp_file_8 = tempfile.NamedTemporaryFile(suffix=".svg", delete=False)
    plt.savefig(temp_file_8.name, format="svg", bbox_inches='tight', pad_inches=0)
    plt.close()

    # 3Dplot-----------------
    nafion_membrane_DF3D = nafion_membrane_DF2.copy()
    # Assuming nafion_membrane_DF3D.values is your matrix data
    data_3d = np.rot90(nafion_membrane_DF3D.values, k=2)

    # Define the range for the Y-axis
    start_y_3d = plot_3d_var[0]
    end_y_3d = plot_3d_var[1]

    # Crop the data to the specified range along the Y-axis
    cropped_data_3d = data_3d[start_y_3d:end_y_3d, :]

    # Create grid for x and y coordinates
    x = np.arange(cropped_data_3d.shape[1], 0, -1)
    y = np.arange(start_y_3d, end_y_3d)  # Adjusted y-coordinates to match the cropped range
    x, y = np.meshgrid(x, y)

    # Create Matplotlib surface plot
    fig = plt.figure(figsize=(6, 4))
    ax = fig.add_subplot(111, projection='3d')
    surf = ax.plot_surface(x, y, cropped_data_3d, cmap='inferno')

    # Set plot title and labels
    ax.set_title('3D plot for intensity values', fontsize=12)
    ax.set_xlabel('X', fontsize=10)
    ax.set_ylabel('Y', fontsize=10)
    ax.set_zlabel('Intensity values (a.u.)', fontsize=10)

    # Set tick parameters
    ax.tick_params(axis='x', labelsize=10)
    ax.tick_params(axis='y', labelsize=10)
    ax.tick_params(axis='z', labelsize=10)
    tick_positions = np.arange(0, cropped_data_3d.shape[1] + 4, 20)  # Set tick positions
    tick_labels = np.arange(cropped_data_3d.shape[1] + 4, 0, -20)  # Set tick labels in reverse order
    ax.set_xticks(tick_positions)
    ax.set_xticklabels(tick_labels)

    # Y axis
    start_value_3d = plot_3d_var[0]
    end_value_3d = plot_3d_var[1]
    # Set Y-axis range
    ax.set_ylim([start_value_3d, end_value_3d])  # Set the desired range
    # Set camera view
    ax.view_init(elev=plot_3d_var[2], azim=plot_3d_var[3])  # Adjust elevation and azimuth angles as needed

    # Save to a temporary file as SVG
    temp_file_9 = tempfile.NamedTemporaryFile(suffix=".svg", delete=False)
    plt.savefig(temp_file_9.name, format="svg", bbox_inches='tight', pad_inches=0)
    plt.close()

    # Otsu plot-----------------

    # Transform nafion_membrane_DF back to array
    nafion_membrane_DF2_1=nafion_membrane_DF2.copy()
    images = nafion_membrane_DF2_1.to_numpy()

    # Applying multi-Otsu threshold for the default value, generating
    # three classes.
    thresholds = threshold_multiotsu(images)

    # Using the threshold values, we generate the three regions.
    regions = np.digitize(images, bins=thresholds)

    plt.rcParams['font.family'] = rcParams
    fig, ax = plt.subplots(figsize=(plot_multi_otsu_var_1[3], plot_multi_otsu_var_1[4]))

    # Plotting the original image.
    p0 = ax.imshow(np.rot90(images, k=2), cmap='gray')
    ax.set_title('Nafion membrane', fontdict=title_font)
    ax.set_xlabel("X (µm)", fontdict=label_font)
    ax.set_ylabel("Y(µm)", fontdict=label_font)
    ax.tick_params(axis='both', labelsize=label_tick)

    # color bar parameters
    color_bar = fig.colorbar(p0, ax=ax,shrink=plot_multi_otsu_var_1[2])
    custom_min_value = plot_multi_otsu_var_1[0]
    custom_max_value = plot_multi_otsu_var_1[1]
    p0.set_clim(vmin=custom_min_value, vmax=custom_max_value)

    plt.subplots_adjust()

    # Save to a temporary file as SVG
    temp_file_10 = tempfile.NamedTemporaryFile(suffix=".svg", delete=False)
    plt.savefig(temp_file_10.name, format="svg", bbox_inches='tight', pad_inches=0)
    plt.close()

    # Histogram---------------
    # Plotting the histogram and the two thresholds obtained from multi-Otsu.
    # Transform nafion_membrane_DF back to array
    nafion_membrane_DF2_2=nafion_membrane_DF2.copy()
    images_2 = nafion_membrane_DF2.to_numpy()

    # Applying multi-Otsu threshold for the default value, generating
    # three classes.

    # Create a subplot and plot the histogram
    n, bins, patches = ax.hist(images_2.ravel(), bins=256, color='blue', alpha=0.7)

    # Normalizing excluding counts for points equal to zero
    n_without_zero = n[1:]
    nonzero_bins = bins[2:]
    count_values = n_without_zero / np.max(n_without_zero)

    plt.rcParams['font.family'] = rcParams


    fig, ax = plt.subplots(figsize=(plot_hist_otsu_var_1[3], plot_hist_otsu_var_1[4]))
    # lines_mem = plt.plot(nonzero_bins, count_values, color="blue",alpha=0.8)

    clipped_nonzero_bins = np.clip(nonzero_bins, plot_hist_otsu_var_1[2], plot_hist_otsu_var_1[1])
    clipped_count_values = np.clip(count_values,0,plot_hist_otsu_var_1[0])

    plt.fill_between(clipped_nonzero_bins, clipped_count_values, color="blue", alpha=0.4)

    ax.set_title('Multi-Otsu averaged thresholds for Nafion membrane', fontdict=title_font)
    ax.set_xlabel("Gray scale values (a.u.)", fontdict=label_font)
    ax.set_ylabel("Frequency (a.u.)", fontdict=label_font)
    ax.set_xlim(plot_hist_otsu_var_1[2], plot_hist_otsu_var_1[1])
    ax.set_ylim(0, plot_hist_otsu_var_1[0])
    ax.tick_params(axis='both', labelsize=label_tick)
    legend_labels = ['Sample']
    legend_elements = [Rectangle((0, 0), 1, 1, color='blue', lw=2)]
    ax.legend(legend_elements, legend_labels, loc=legend_loc, fontsize=legend_size)
    # ax.plot(nonzero_bins, nonzero_values, color='red', marker='o', linestyle='-', linewidth=2)

    for thresh in thresholds:
        if plot_hist_otsu_var_1[2] <= thresh <= plot_hist_otsu_var_1[1]:
            ax.axvline(thresh, color='r')

    plt.subplots_adjust()

    # Save to a temporary file as SVG
    temp_file_11 = tempfile.NamedTemporaryFile(suffix=".svg", delete=False)
    plt.savefig(temp_file_11.name, format="svg", bbox_inches='tight', pad_inches=0)
    plt.close()

    return temp_file_8.name,nafion_membrane_DF2,temp_file_9.name,temp_file_10.name,temp_file_11.name


def segment_nafion(address_final_plots,address_csv1,address_csv2, element,acronym, detector, nafion_membrane_DF2, plot_main_var_1,plot_3d_var_2, plot_main_multi_otsu_var_1, plot_main_histogram_var_1,legend_loc_1,legend_loc_2):

    # Directories
    path_pic = f"{address_final_plots}Final_Plots_paper_{element}_det_{detector}/"

    ## load pristine histogram
    nonzero_bins_gray_pris = np.genfromtxt(address_csv1 ,delimiter=',', skip_header=1)
    normalized_histogram_gray_pris = np.genfromtxt(address_csv2 ,delimiter=',', skip_header=1)


    ## Final Plots
    plt.rcParams['font.family'] = rcParams

    nafion_membrane_DF2_1 = nafion_membrane_DF2.copy()
    images = nafion_membrane_DF2_1.to_numpy()
    images_1 = images.copy()
    images_2 = images.copy()


    fig, ax = plt.subplots(nrows=1, ncols=4, figsize=(plot_main_var_1[0], plot_main_var_1[1]))

    # Plotting the original image.----------------------------------------------------------

    ax1 = ax[0]
    p1 = ax1.imshow(np.rot90(images, k=2), cmap='gray')
    # ax1.set_title('Nafion membrane', fontdict=title_font)
    ax1.set_xlabel("X (µm)", fontdict=label_font)
    ax1.set_ylabel("Y(µm)", fontdict=label_font)
    ax1.tick_params(axis='both', labelsize=label_tick)
    ax1.set_ylim(130, 0)  # Set y-axis limits for the second subplot

    # Create a color bar inside the image
    color_bar = fig.colorbar(p1, ax=ax1, shrink=plot_main_multi_otsu_var_1[2], pad=0.05)
    color_bar.set_label('Gray scale values (a.u.)', fontsize=10, family='Arial')
    custom_min_value = plot_main_multi_otsu_var_1[0]
    custom_max_value = plot_main_multi_otsu_var_1[1]
    p1.set_clim(vmin=custom_min_value, vmax=custom_max_value)


    # Plotting 3D ---------------------------------------------------------------
    # Define the range for the Y-axis

    start_y_3d = plot_3d_var_2[0]
    end_y_3d =  plot_3d_var_2[1]

    # Crop the data to the specified range along the Y-axis
    nafion_membrane_DF2_2 = nafion_membrane_DF2.copy()
    data_3d = np.rot90(nafion_membrane_DF2_2.values, k=2)
    cropped_data_3d = data_3d[start_y_3d:end_y_3d, :]

    # Create grid for x and y coordinates
    x = np.arange(cropped_data_3d.shape[1], 0, -1)
    y = np.arange(start_y_3d, end_y_3d)  # Adjusted y-coordinates to match the cropped range
    x, y = np.meshgrid(x, y)

    # Create Matplotlib surface plot
    ax0 = ax[1]
    ax0.axis('off')
    ax0 = fig.add_subplot(142, projection='3d')
    surf = ax0.plot_surface(x, y, cropped_data_3d, cmap='inferno')

    # Set plot title and labels
    # ax0.set_title('3D plot for intensity values', fontsize=12)
    ax0.set_xlabel('X (µm)', fontsize=10)
    ax0.set_ylabel('Y (µm)', fontsize=10)
    ax0.set_zlabel('Intensity values (a.u.)', fontsize=10)

    # Set tick parameters
    ax0.tick_params(axis='x', labelsize=10)
    ax0.tick_params(axis='y', labelsize=10)
    ax0.tick_params(axis='z', labelsize=10)

    tick_positions = np.arange(0, cropped_data_3d.shape[1] + 4, 20)  # Set tick positions
    tick_labels = np.arange(cropped_data_3d.shape[1] + 4, 0, -20)  # Set tick labels in reverse order
    ax0.set_xticks(tick_positions)
    ax0.set_xticklabels(tick_labels)

    # Y axis
    start_value_3d = 0
    end_value_3d = 140
    # Set Y-axis range
    ax0.set_ylim([start_value_3d, end_value_3d])  # Set the desired range
    # Set camera view
    ax0.view_init(elev=plot_3d_var_2[2], azim=plot_3d_var_2[3])  # Adjust elevation and azimuth angles as needed
    fig.patch.set_facecolor('none')
    ax0.set_facecolor('none')

    # Histograms--------------------------------------------------------------------------

    # Create a subplot and plot the histogram
    n, bins = np.histogram(images_1.ravel(), bins=256)
    # Normalizing excluding counts for points equal to zero
    n_without_zero = n[1:]
    nonzero_bins = bins[2:]
    count_values = n_without_zero / np.max(n_without_zero)

    ax2 = ax[2]
    xlim_min, xlim_max = plot_main_histogram_var_1[0], plot_main_histogram_var_1[1]
    ylim_min, ylim_max = 0, plot_main_histogram_var_1[2]

    clipped_nonzero_bins_gray_pris = np.clip(nonzero_bins_gray_pris, xlim_min, xlim_max)
    clipped_normalized_histogram_gray_pris = np.clip(normalized_histogram_gray_pris, ylim_min, ylim_max)

    clipped_nonzero_bins_sample = np.clip(nonzero_bins, xlim_min, xlim_max)
    clipped_normalized_histogram_sample = np.clip(count_values, ylim_min, ylim_max)

    ax2.fill_between(clipped_nonzero_bins_gray_pris,  clipped_normalized_histogram_gray_pris, color="blue", alpha=0.4)
    ax2.fill_between(clipped_nonzero_bins_sample, clipped_normalized_histogram_sample, color="orchid", alpha=0.6)
    ax2.set_xlabel("Gray scale values (a.u.)", fontdict=label_font)
    ax2.set_ylabel("Normalized frequency (a.u.)", fontdict=label_font)
    ax2.tick_params(axis='both', labelsize=label_tick)
    ax2.set_xlim(xlim_min, xlim_max)
    ax2.set_ylim(ylim_min, ylim_max)  # change according to analysis

    thresholds = threshold_multiotsu(images_1)
    for thresh in thresholds:
        if xlim_min <= thresh <= xlim_max:
            ax2.axvline(thresh, color='r')

    legend_labels = [acronym, 'Pristine CCM', 'Threshold']
    legend_elements = [Rectangle((0, 0), 1, 1, color='orchid', lw=2),
                       Rectangle((0, 0), 1, 1, color='blue', lw=2),
                       Rectangle((0, 0), 1, 1, color='red', lw=2)]
    ax2.legend(legend_elements, legend_labels, loc=legend_loc_1, fontsize=legend_size)


    # Segmented image--------------------------------------------------------------------------

    regions = np.digitize(images_2, bins=thresholds)

    ax3 = ax[3]
    p3 = ax3.imshow(np.rot90(regions, k=2), cmap='inferno')
    # ax3.set_title('Segmented Nafion', fontdict=title_font)
    ax3.set_xlabel("X (µm)", fontdict=label_font)
    ax3.set_ylabel("Y(µm)", fontdict=label_font)
    ax3.tick_params(axis='both', labelsize=label_tick)
    legend_labels = [element, 'Nafion']
    legend_elements = [Rectangle((0, 0), 1, 1, color='#FCFFA4', lw=2),
                       Rectangle((0, 0), 1, 1, color='#D44842', lw=2)]
    ax3.legend(legend_elements, legend_labels, loc=legend_loc_2)
    ax3.set_ylim(130, 0)  # Set y-axis limits for the second subplot

    # Adjust horizontal spacing and centralize each subplot
    fig.subplots_adjust(wspace=0.3)


    # Customize the file path
    file_path = path_pic + 'Main2.png'

    # Save only the first subplot
    fig.savefig(file_path, dpi=dpi_num)

    # Save to a temporary file as SVG
    temp_file_12 = tempfile.NamedTemporaryFile(suffix=".svg", delete=False)
    plt.savefig(temp_file_12.name, format="svg", bbox_inches='tight', pad_inches=0)
    plt.close()

    return temp_file_12.name, images


def thickness(address_final_plots,address_single_tiff3,element,acronym, detector, nafion_membrane_DF1,gray_scale_normal_plot,thickness_normal_plot,gray_scale_histogram_plot,thickness_histogram_plot,legend_loc_var,main_parameters):

    #Directories
    path_pic = f"{address_final_plots}Final_Plots_paper_{element}_det_{detector}/"

    # load the real processed image
    uri3 = address_single_tiff3
    image = io.imread(uri3)
    im = image.copy()
    gray_values_abs = np.rot90(im, k=2)

    ##------------Get only membrane absorption values------------##

    gray_values_abs_DF = pd.DataFrame(gray_values_abs)
    gray_values_abs_DF = gray_values_abs_DF.fillna(0)

    # Find coordinates of pixels with value 0 in image1
    coordinates = np.argwhere(nafion_membrane_DF1.values == 0) # DF2 is also possible, but if we have dust in the middle, then we will have thickness = o and later or quanti x/0 is not possible

    # Set corresponding pixels to 0 in image2
    for coord in coordinates:
        gray_values_abs_DF.iloc[coord[0], coord[1]] = 0

    ##----------------------Data treatment-----------------------##

    ## Blur to data
    gray_values_abs_array = gray_values_abs_DF.values
    # blurry data for segmentation and generating threshold value
    blurred_values_abs_array = ski.filters.gaussian(gray_values_abs_array, sigma=0.8)

    # Create the matrix filled with the specified value
    rows_sample, columns_sample = gray_values_abs_array.shape
    air_mean_value =  main_parameters[0]
    matrix_air = np.full((rows_sample, columns_sample), air_mean_value)

    ##Converting gray_values of absorption contrast to Thickness
    gray_values_thickness = (gray_values_abs_array - matrix_air) *  main_parameters[1]  # X-Ray attenuation lenght for Nafion
    gray_values_thickness_DF = pd.DataFrame(gray_values_thickness)
    binary_mask_values_thickness_DF = gray_values_thickness_DF > 0
    gray_values_thickness_DF[~binary_mask_values_thickness_DF] = 0

    ##-----------------------Histograms-------------------------##

    ##Histogram for abs signal
    filtered_data_abs = gray_values_abs_DF.values[gray_values_abs_DF.values != 0]
    histogram_abs, bin_edges_abs = np.histogram(filtered_data_abs, bins=256)
    # Find the maximum count in the histogram
    max_count_abs = np.max(histogram_abs)
    # Normalize the histogram data
    normalized_histogram_abs = histogram_abs / max_count_abs

    ##Histogram for thickness
    filtered_data_thickness = gray_values_thickness_DF.values[gray_values_thickness_DF.values != 0]
    histogram, bin_edges = np.histogram(filtered_data_thickness, bins=256)
    # Find the maximum count in the histogram
    max_count = np.max(histogram)
    # Normalize the histogram data
    normalized_histogram = histogram / max_count

    ##--------------------------Plots--------------------------##

    plt.rcParams['font.family'] = rcParams

    fig, ax = plt.subplots(nrows=2, ncols=2, figsize=(main_parameters[2], main_parameters[3]))

    ###### Gray scale values
    ax1 = plt.subplot(2, 2, 1)
    p1 = ax1.imshow(np.rot90(gray_values_abs_DF, k=2), cmap='gray')
    ax1.set_title('Nafion and epoxy', fontdict=title_font)
    ax1.set_xlabel('X (µm)', fontdict=label_font)
    ax1.set_ylabel('Y (µm)', fontdict=label_font)
    color_bar = fig.colorbar(p1, ax=ax1, shrink=gray_scale_normal_plot[1], pad=0.05)
    color_bar.set_label('Gray scale values (a.u.)', fontdict=label_font)
    color_bar.set_label('Gray scale values (a.u.)', fontdict=label_font)
    custom_min_value = gray_scale_normal_plot[0]
    custom_max_value = gray_scale_normal_plot[2]
    p1.set_clim(vmin=custom_min_value, vmax=custom_max_value)

    ###### Histogram
    clipped_bin_edges_abs = np.clip(bin_edges_abs, gray_scale_histogram_plot[1], gray_scale_histogram_plot[2])
    clipped_normalized_histogram_abs = np.clip(normalized_histogram_abs, 0,gray_scale_histogram_plot[0])

    ax2 = plt.subplot(2, 2, 2)
    ax2.fill_between(clipped_bin_edges_abs[0:-1], clipped_normalized_histogram_abs, color="blue", alpha=0.4)
    ax2.set_xlabel("Gray scale values (a.u.)", fontdict=label_font)
    ax2.set_ylabel("Normalized frequency (a.u.)", fontdict=label_font)
    ax2.set_xlim(gray_scale_histogram_plot[1], gray_scale_histogram_plot[2])
    ax2.set_ylim(0, gray_scale_histogram_plot[0])  # change according to analysis
    legend_labels = [acronym]
    legend_elements = [Rectangle((0, 0), 1, 1, color='blue', lw=2)]
    ax2.legend(legend_elements, legend_labels, loc=legend_loc_var[0], fontsize=legend_size)

    ###### Thickness
    ax3 = plt.subplot(2, 2, 3)
    p3 = ax3.imshow(np.rot90(gray_values_thickness_DF, k=2), cmap='inferno')
    ax3.set_title('Nafion and epoxy', fontdict=title_font)
    ax3.set_xlabel('X (µm)', fontdict=label_font)
    ax3.set_ylabel('Y (µm)', fontdict=label_font)
    color_bar = fig.colorbar(p3, ax=ax3, shrink=thickness_normal_plot[2], pad=0.05)
    color_bar.set_label('Thickness (μm)', fontdict=label_font)
    custom_min_value_p3 = thickness_normal_plot[0]
    custom_max_value_p3 = thickness_normal_plot[1]
    p3.set_clim(vmin=custom_min_value_p3, vmax=custom_max_value_p3)

    ###### Histogram
    clipped_bin_edges_thickness = np.clip(bin_edges, thickness_histogram_plot[2], thickness_histogram_plot[1])
    clipped_normalized_histogram_thickness = np.clip(normalized_histogram, 0, thickness_histogram_plot[0])

    ax4 = plt.subplot(2, 2, 4)
    ax4.fill_between(clipped_bin_edges_thickness[0:-1], clipped_normalized_histogram_thickness, color="orange", alpha=0.4)
    ax4.set_xlabel("Thickness (μm)", fontdict=label_font)
    ax4.set_ylabel("Normalized frequency (a.u.)", fontdict=label_font)
    ax4.set_xlim(thickness_histogram_plot[2], thickness_histogram_plot[1])
    ax4.set_ylim(0, thickness_histogram_plot[0])  # change according to analysis
    legend_labels = [acronym]
    legend_elements = [Rectangle((0, 0), 1, 1, color='orange', lw=2)]
    ax4.legend(legend_elements, legend_labels, loc=legend_loc_var[1], fontsize=legend_size)

    plt.tight_layout()

    # Customize the file path
    file_path = path_pic + 'Thickness_map.png'

    # Save only the first subplot
    fig.savefig(file_path, dpi=dpi_num)


    # Save to a temporary file as SVG
    temp_file_13 = tempfile.NamedTemporaryFile(suffix=".svg", delete=False)
    plt.savefig(temp_file_13.name, format="svg", bbox_inches='tight', pad_inches=0)
    plt.close()

    return temp_file_13.name, gray_values_thickness_DF


def quantify(address_final_plots,address_csv3,address_csv4,detector,element,acronym,images,gray_values_thickness_DF,figure_parameters_quantify,figure_3dplot_quantify,figure_histogram_quantify,figure_membrane_quantify,legend_loc_quantify):

    # Directories
    path_pic = f"{address_final_plots}Final_Plots_paper_{element}_det_{detector}/"

    images_DF = pd.DataFrame(images)
    quanti_DF = images_DF * figure_parameters_quantify[2] / (gray_values_thickness_DF * 1E-4)
    quanti_DF = quanti_DF.replace([np.inf, -np.inf], np.nan)  ## weird numbers appearing (due to number/0 tends to inf)
    quanti_DF.fillna(0, inplace=True)  ## removes 0's
    quanti = quanti_DF.values  ## change to numpy array

    ## load pristine sample data for histogram

    bin_edges_quanti_pris = np.genfromtxt(address_csv3, delimiter=',', skip_header=1)
    normalized_histogram_quanti_pris = np.genfromtxt(address_csv4, delimiter=',', skip_header=1)

    ##histogram for quantities
    filtered_data_quanti = quanti[quanti != 0]
    histogram_quanti, bin_edges_quanti = np.histogram(filtered_data_quanti, bins=256)

    bin_edges_quanti = bin_edges_quanti[1:]
    histogram_quanti = histogram_quanti[1:]

    # Find the maximum count in the histogram
    max_count_quanti = np.max(histogram_quanti)
    # Normalize the histogram data
    normalized_histogram_quanti = histogram_quanti / max_count_quanti

    ##--------------------------Plots--------------------------##
    plt.rcParams['font.family'] = rcParams

    fig, ax = plt.subplots(nrows=1, ncols=3, figsize=(figure_parameters_quantify[1], figure_parameters_quantify[0]))

    ax1 = plt.subplot(1, 3, 1)
    p1 = ax1.imshow(np.rot90(quanti_DF, k=2), cmap='inferno')
    ax1.set_xlabel('X (µm)', fontdict=label_font)
    ax1.set_ylabel('Y (µm)', fontdict=label_font)
    color_bar = fig.colorbar(p1, ax=ax1, shrink=figure_membrane_quantify[1], pad=0.05)
    color_bar.set_label('Quantity (atoms/cm³)', fontdict=label_font)
    custom_min_value = figure_membrane_quantify[0]
    custom_max_value = figure_membrane_quantify[2]
    p1.set_clim(vmin=custom_min_value, vmax=custom_max_value)


    ###### Histogram

    clipped_bin_edges_quanti_pris = np.clip(bin_edges_quanti_pris, figure_histogram_quantify[2],figure_histogram_quantify[1])
    clipped_normalized_histogram_quanti_pris = np.clip(normalized_histogram_quanti_pris, 0, figure_histogram_quantify[0])

    clipped_bin_edges_quanti = np.clip(bin_edges_quanti, figure_histogram_quantify[2], figure_histogram_quantify[1])
    clipped_normalized_histogram_quanti = np.clip(normalized_histogram_quanti, 0, figure_histogram_quantify[0])

    ax2 = plt.subplot(1, 3, 2)
    ax2.fill_between(clipped_bin_edges_quanti_pris[0:-1], clipped_normalized_histogram_quanti_pris, color="blue", alpha=0.4)
    ax2.fill_between(clipped_bin_edges_quanti[0:-1], clipped_normalized_histogram_quanti, color="orchid", alpha=0.8)
    ax2.set_xlabel("Quantity (atoms/cm³)", fontdict=label_font)
    ax2.set_ylabel("Normalized frequency (a.u.)", fontdict=label_font)
    ax2.set_xlim(figure_histogram_quantify[2],figure_histogram_quantify[1])
    ax2.set_ylim(0, figure_histogram_quantify[0])  # change according to analysis
    legend_labels = [acronym, 'Pristine CCM']
    legend_elements = [Rectangle((0, 0), 1, 1, color='orchid', lw=2),
                       Rectangle((0, 0), 1, 1, color='blue', lw=2)]
    ax2.legend(legend_elements, legend_labels, loc=legend_loc_quantify, fontsize=legend_size)
    fig.patch.set_facecolor('none')
    ax2.set_facecolor('none')


    # Plotting 3D ---------------------------------------------------------------
    # Assuming nafion_membrane_DF3D.values is your matrix data
    data_3d_thick = np.rot90(quanti, k=2)
    # Define the range for the Y-axis
    start_y_3d_thick = figure_3dplot_quantify[0]
    end_y_3d_thick = figure_3dplot_quantify[1]

    # Crop the data to the specified range along the Y-axis
    cropped_data_3d_thick = data_3d_thick[start_y_3d_thick:end_y_3d_thick, :]

    # Create grid for x and y coordinates
    x_thick = np.arange(cropped_data_3d_thick.shape[1], 0, -1)
    y_thick = np.arange(start_y_3d_thick, end_y_3d_thick)  # Adjusted y-coordinates to match the cropped range
    x_thick, y_thick = np.meshgrid(x_thick, y_thick)

    # Create Matplotlib surface plot
    ax3 = plt.subplot(1, 3, 3)
    ax3.axis('off')
    ax3 = fig.add_subplot(133, projection='3d')
    surf_thick = ax3.plot_surface(x_thick, y_thick, cropped_data_3d_thick, cmap='inferno')

    # Set plot title and labels
    # ax0.set_title('3D plot for intensity values', fontsize=12)
    ax3.set_xlabel('X (µm)', fontdict=label_font)
    ax3.set_ylabel('Y (µm)', fontdict=label_font)
    ax3.set_zlabel('', fontdict=label_font, labelpad=-1)

    # Custom formatter to remove the + sign from scientific notation
    def scientific_formatter(x, pos):
        if x == 0:
            return '0'
        # Convert to scientific notation with no + sign for the exponent
        return f'{x:.1e}'.replace('e+0', 'e0').replace('e+0', 'e0').replace('e+', 'e').replace('e-0', 'e-')

    # Apply custom formatter to the z-axis
    ax3.zaxis.set_major_formatter(FuncFormatter(scientific_formatter))

    # Custom annotation for z-axis label
    ax3.text2D(-0.2, 0.25, 'Quantity (atoms/cm³)', fontdict=label_font, rotation=96, transform=ax3.transAxes)

    # Set tick parameters
    ax3.tick_params(axis='x', labelsize=10)
    ax3.tick_params(axis='y', labelsize=10)
    ax3.tick_params(axis='z', labelsize=10)

    tick_positions_thick = np.arange(0, cropped_data_3d_thick.shape[1] + figure_3dplot_quantify[4], 20)  # Set tick positions
    tick_labels_thick = np.arange(cropped_data_3d_thick.shape[1] + figure_3dplot_quantify[4], 0, -20)  # Set tick labels in reverse order
    ax3.set_xticks(tick_positions_thick)
    ax3.set_xticklabels(tick_labels_thick)

    # Y axis
    start_value_3d_thick = figure_3dplot_quantify[0]
    end_value_3d_thick = figure_3dplot_quantify[1]
    # Set Y-axis range
    ax3.set_ylim([start_value_3d_thick, end_value_3d_thick])  # Set the desired range
    # Set camera view
    ax3.view_init(elev=figure_3dplot_quantify[2] , azim=figure_3dplot_quantify[3])  # Adjust elevation and azimuth angles as needed
    # remove white background
    fig.patch.set_facecolor('none')
    ax3.set_facecolor('none')

    # Customize the file path
    file_path = path_pic + 'Quanti.png'
    # Save only the first subplot
    fig.savefig(file_path, dpi=dpi_num)

    # Save to a temporary file as SVG
    temp_file_14 = tempfile.NamedTemporaryFile(suffix=".svg", delete=False)
    plt.savefig(temp_file_14.name, format="svg", bbox_inches='tight', pad_inches=0)
    plt.close()

    return temp_file_14.name


class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(1122, 1046)

        # Determine the correct path for the logo based on whether we are running from a bundled exe
        if getattr(sys, 'frozen', False):
            # Running as a bundled exe
            logo_path = sys._MEIPASS + '/resources/logo.png'
        else:
            # Running as a script (development mode)
            logo_path = 'resources/logo.png'

        # Set the window icon
        icon = QtGui.QIcon(logo_path)
        MainWindow.setWindowIcon(icon)

        self.centralwidget = QtWidgets.QWidget(parent=MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.tabWidget = QtWidgets.QTabWidget(parent=self.centralwidget)
        self.tabWidget.setGeometry(QtCore.QRect(0, 0, 1121, 1001))
        self.tabWidget.setObjectName("tabWidget")

        # Tab 1---------------------------------------------------------------------------------------------------------
        # region
        self.tab = QtWidgets.QWidget()
        self.tab.setObjectName("tab")

        # Use QSpinBox for "Day"
        self.spinBox_2 = QtWidgets.QSpinBox(parent=self.tab)
        self.spinBox_2.setGeometry(QtCore.QRect(10, 30, 113, 22))
        self.spinBox_2.setObjectName("spinBox_2")
        self.spinBox_2.setRange(10000000, 99999999)  # Set the valid range (e.g., 8-digit number)
        self.spinBox_2.setValue(20230623)  # Set initial value for "Day"

        # Prefill "Figure size x" with default text
        self.lineEdit_1 = QtWidgets.QLineEdit(parent=self.tab)
        self.lineEdit_1.setGeometry(QtCore.QRect(220, 200, 113, 22))
        self.lineEdit_1.setText("8")  # Set the initial text to "Detector_1"
        self.lineEdit_1.setEnabled(False)
        self.lineEdit_1.setObjectName("lineEdit_1")

        # Prefill "Figure size y" with default text
        self.lineEdit_2 = QtWidgets.QLineEdit(parent=self.tab)
        self.lineEdit_2.setGeometry(QtCore.QRect(420, 200, 113, 22))
        self.lineEdit_2.setText("6")  # Set the initial text to "Detector_1"
        self.lineEdit_2.setEnabled(False)
        self.lineEdit_2.setObjectName("lineEdit_2")

        # Prefill "Color bar shrink" with default text
        self.lineEdit_3 = QtWidgets.QLineEdit(parent=self.tab)
        self.lineEdit_3.setGeometry(QtCore.QRect(670, 200, 113, 22))
        self.lineEdit_3.setText("0.8")  # Set the initial text to "Detector_1"
        self.lineEdit_3.setEnabled(False)
        self.lineEdit_3.setObjectName("lineEdit_3")

        # Prefill "Sample_name" with default text
        self.lineEdit_4 = QtWidgets.QLineEdit(parent=self.tab)
        self.lineEdit_4.setGeometry(QtCore.QRect(10, 60, 113, 22))
        self.lineEdit_4.setText(
            "20230623194043_noXRD_XRF_XY_100ms_11700eV_015_100h_20_01_detail_hc")  # Set the initial text to "Sample_123"
        self.lineEdit_4.setObjectName("lineEdit_4")

        # Prefill "Detector" with default text
        self.lineEdit_5 = QtWidgets.QLineEdit(parent=self.tab)
        self.lineEdit_5.setGeometry(QtCore.QRect(10, 90, 113, 22))
        self.lineEdit_5.setText("02")  # Set the initial text to "Detector_1"
        self.lineEdit_5.setObjectName("lineEdit_5")

        # Sample Acronym
        self.lineEdit_70 = QtWidgets.QLineEdit(parent=self.tab)
        self.lineEdit_70.setGeometry(QtCore.QRect(220, 60, 113, 22))
        self.lineEdit_70.setText("100 h CCM")  # Set the initial text to "Detector_1"
        self.lineEdit_70.setObjectName("lineEdit_70")

        # Change "Element" to a QComboBox
        self.comboBox_element = QtWidgets.QComboBox(parent=self.tab)
        self.comboBox_element.setGeometry(QtCore.QRect(10, 120, 113, 22))
        self.comboBox_element.setObjectName("comboBox_element")
        # Deactivate the combo box initially
        self.comboBox_element.setEnabled(False)
        self.comboBox_element.addItem("-")

        self.label = QtWidgets.QLabel(parent=self.tab)
        self.label.setGeometry(QtCore.QRect(10, 10, 71, 16))
        self.label.setObjectName("label")

        self.label_1 = QtWidgets.QLabel(parent=self.tab)
        self.label_1.setGeometry(QtCore.QRect(340, 200, 81, 21))
        self.label_1.setObjectName("label_1")

        self.label_1_1 = QtWidgets.QLabel(parent=self.tab)
        self.label_1_1.setGeometry(QtCore.QRect(540, 200, 81, 21))
        self.label_1_1.setObjectName("label_1_1")

        self.label_1_2 = QtWidgets.QLabel(parent=self.tab)
        self.label_1_2.setGeometry(QtCore.QRect(790, 200, 121, 21))
        self.label_1_2.setObjectName("label_1_2")

        self.label_1_3 = QtWidgets.QLabel(parent=self.tab)
        self.label_1_3.setGeometry(QtCore.QRect(60, 760, 61, 21))
        self.label_1_3.setObjectName("label_1_3")

        self.label_1_4 = QtWidgets.QLabel(parent=self.tab)
        self.label_1_4.setGeometry(QtCore.QRect(340, 60, 81, 16))
        self.label_1_4.setObjectName("label_1_4")

        self.label_2 = QtWidgets.QLabel(parent=self.tab)
        self.label_2.setGeometry(QtCore.QRect(130, 30, 71, 16))
        self.label_2.setObjectName("label_2")

        self.label_3 = QtWidgets.QLabel(parent=self.tab)
        self.label_3.setGeometry(QtCore.QRect(130, 60, 81, 16))
        self.label_3.setObjectName("label_3")

        self.label_4 = QtWidgets.QLabel(parent=self.tab)
        self.label_4.setGeometry(QtCore.QRect(130, 90, 49, 16))
        self.label_4.setObjectName("label_4")

        self.label_5 = QtWidgets.QLabel(parent=self.tab)
        self.label_5.setGeometry(QtCore.QRect(130, 120, 49, 16))
        self.label_5.setObjectName("label_5")

        # pushButton_start_project
        self.project_path = None # folder to save project
        self.address_stack_tiff1 = None # Stack from sum of detectors
        self.address_stack_tiff2 = None  # Stack from single detector
        self.address_single_tiff3 = None  # mux_roi tiff
        self.address_plot_of_stacks = None
        self.address_final_plots = None
        self.pushButton_start_project = QtWidgets.QPushButton(parent=self.tab)
        self.pushButton_start_project.setGeometry(QtCore.QRect(830, 30, 121, 24))
        self.pushButton_start_project.setObjectName("pushButton_start_project")
        self.pushButton_start_project.clicked.connect(self.handle_start_project)

        self.listed_elements = None
        self.pushButton = QtWidgets.QPushButton(parent=self.tab)
        self.pushButton.setGeometry(QtCore.QRect(220, 30, 75, 24))
        self.pushButton.setObjectName("pushButton")
        self.pushButton.setEnabled(False)
        self.pushButton.clicked.connect(self.handle_import_stack_names)

        # Pushbutton_Import
        # Initialize instance variables for gray values
        self.gray_values_pt = None
        self.gray_values_ir = None
        self.pushButton_2 = QtWidgets.QPushButton(parent=self.tab)
        self.pushButton_2.setGeometry(QtCore.QRect(220, 120, 75, 24))
        self.pushButton_2.setObjectName("pushButton_2")
        self.pushButton_2.clicked.connect(self.handle_import_file)
        self.pushButton_2.setEnabled(False)

        self.pushButton_12 = QtWidgets.QPushButton(parent=self.tab)
        self.pushButton_12.setGeometry(QtCore.QRect(50, 800, 75, 24))
        self.pushButton_12.setObjectName("pushButton_12")
        self.pushButton_12.clicked.connect(self.handle_save_varibles)

        self.filename = None
        self.pushButton_13 = QtWidgets.QPushButton(parent=self.tab)
        self.pushButton_13.setGeometry(QtCore.QRect(50, 860, 75, 24))
        self.pushButton_13.setObjectName("pushButton_13")
        self.pushButton_13.clicked.connect(self.handle_load_variables)


        self.pushButton_clean = QtWidgets.QPushButton(parent=self.tab)
        self.pushButton_clean.setGeometry(QtCore.QRect(830, 860, 121, 24))
        self.pushButton_clean.setObjectName("pushButton_clean")
        self.pushButton_clean.setEnabled(False)
        self.pushButton_clean.clicked.connect(self.handle_clear_all_graphics_views)

        self.graphicsView = QtWidgets.QGraphicsView(parent=self.tab)
        self.graphicsView.setGeometry(QtCore.QRect(180, 230, 771, 471))
        self.graphicsView.setObjectName("graphicsView")

        self.line_21 = QtWidgets.QFrame(parent=self.tab)
        self.line_21.setGeometry(QtCore.QRect(30, 740, 118, 3))
        self.line_21.setFrameShape(QtWidgets.QFrame.Shape.HLine)
        self.line_21.setFrameShadow(QtWidgets.QFrame.Shadow.Sunken)
        self.line_21.setObjectName("line_21")

        self.line_22 = QtWidgets.QFrame(parent=self.tab)
        self.line_22.setGeometry(QtCore.QRect(30, 910, 118, 3))
        self.line_22.setFrameShape(QtWidgets.QFrame.Shape.HLine)
        self.line_22.setFrameShadow(QtWidgets.QFrame.Shadow.Sunken)
        self.line_22.setObjectName("line_22")

        self.line_23 = QtWidgets.QFrame(parent=self.tab)
        self.line_23.setGeometry(QtCore.QRect(150, 750, 20, 151))
        self.line_23.setFrameShape(QtWidgets.QFrame.Shape.VLine)
        self.line_23.setFrameShadow(QtWidgets.QFrame.Shadow.Sunken)
        self.line_23.setObjectName("line_23")

        self.line_24 = QtWidgets.QFrame(parent=self.tab)
        self.line_24.setGeometry(QtCore.QRect(10, 750, 20, 151))
        self.line_24.setFrameShape(QtWidgets.QFrame.Shape.VLine)
        self.line_24.setFrameShadow(QtWidgets.QFrame.Shadow.Sunken)
        self.line_24.setObjectName("line_24")


        self.tabWidget.addTab(self.tab, "")  # End tab
        # endregion

        # Tab 2---------------------------------------------------------------------------------------------------------
        # region
        self.tab_2 = QtWidgets.QWidget()
        self.tab_2.setObjectName("tab_2")

        self.comboBox_methods_thresh = QtWidgets.QComboBox(parent=self.tab_2)
        self.comboBox_methods_thresh.setGeometry(QtCore.QRect(20, 20, 101, 22))
        self.comboBox_methods_thresh.setObjectName("comboBox_methods_thresh")
        methods_thresh = ['Manual', 'Otsu', 'Minimum', 'Triangle']
        for style_sel in methods_thresh:
            self.comboBox_methods_thresh.addItem(style_sel)  # Add each modified filename to the combo box

        self.comboBox_methods_thresh.setEnabled(False)


        # Pushbutton_3
        # Initialize instance variables for gray values
        self.val_pt = None
        self.val_ir = None
        self.blurred_gray_pt = None
        self.blurred_gray_ir = None

        self.pushButton_3 = QtWidgets.QPushButton(parent=self.tab_2)
        self.pushButton_3.setGeometry(QtCore.QRect(660, 130, 211, 24))
        self.pushButton_3.setObjectName("pushButton_3")
        self.pushButton_3.setEnabled(False)
        self.pushButton_3.clicked.connect(self.handle_thresholding_blur)

        # Pt manual threshold
        self.lineEdit_6 = QtWidgets.QLineEdit(parent=self.tab_2)
        self.lineEdit_6.setGeometry(QtCore.QRect(20, 80, 113, 22))
        self.lineEdit_6.setObjectName("lineEdit_6")
        self.lineEdit_6.setEnabled(False)
        self.lineEdit_6.setText("100")

        # Ir manual threshold
        self.lineEdit_7 = QtWidgets.QLineEdit(parent=self.tab_2)
        self.lineEdit_7.setGeometry(QtCore.QRect(20, 110, 113, 22))
        self.lineEdit_7.setObjectName("lineEdit_7")
        self.lineEdit_7.setEnabled(False)
        self.lineEdit_7.setText("1000")

        self.label_6 = QtWidgets.QLabel(parent=self.tab_2)
        self.label_6.setGeometry(QtCore.QRect(130, 20, 101, 16))
        self.label_6.setObjectName("label_6")

        self.label_7 = QtWidgets.QLabel(parent=self.tab_2)
        self.label_7.setGeometry(QtCore.QRect(140, 80, 101, 16))
        self.label_7.setObjectName("label_7")

        self.label_8 = QtWidgets.QLabel(parent=self.tab_2)
        self.label_8.setGeometry(QtCore.QRect(140, 110, 101, 16))
        self.label_8.setObjectName("label_8")

        self.label_9 = QtWidgets.QLabel(parent=self.tab_2)
        self.label_9.setGeometry(QtCore.QRect(70, 50, 111, 20))
        self.label_9.setObjectName("label_9")

        self.label_10 = QtWidgets.QLabel(parent=self.tab_2)
        self.label_10.setGeometry(QtCore.QRect(20, 150, 111, 20))
        self.label_10.setObjectName("label_10")

        self.label_11 = QtWidgets.QLabel(parent=self.tab_2)
        self.label_11.setGeometry(QtCore.QRect(20, 540, 111, 20))
        self.label_11.setObjectName("label_11")

        # main panel
        self.graphicsView_2 = QtWidgets.QGraphicsView(parent=self.tab_2)
        self.graphicsView_2.setGeometry(QtCore.QRect(430, 170, 671, 731))
        self.graphicsView_2.setObjectName("graphicsView_2")
        # dialog box
        self.graphicsView_3 = QtWidgets.QGraphicsView(parent=self.tab_2)
        self.graphicsView_3.setGeometry(QtCore.QRect(500, 910, 541, 31))
        self.graphicsView_3.setObjectName("graphicsView_3")

        # Pt histogram
        self.graphicsView_4 = QtWidgets.QGraphicsView(parent=self.tab_2)
        self.graphicsView_4.setGeometry(QtCore.QRect(20, 170, 391, 341))
        self.graphicsView_4.setObjectName("graphicsView_4")

        # Ir histogram
        self.graphicsView_5 = QtWidgets.QGraphicsView(parent=self.tab_2)
        self.graphicsView_5.setGeometry(QtCore.QRect(20, 560, 391, 341))
        self.graphicsView_5.setObjectName("graphicsView_5")

        self.tabWidget.addTab(self.tab_2, "")

        # endregion

        # Tab 3---------------------------------------------------------------------------------------------------------
        # region
        self.tab_3 = QtWidgets.QWidget()
        self.tab_3.setObjectName("tab_3")

        # Figure size x
        self.lineEdit_8 = QtWidgets.QLineEdit(parent=self.tab_3)
        self.lineEdit_8.setGeometry(QtCore.QRect(330, 170, 113, 22))
        self.lineEdit_8.setObjectName("lineEdit_8")
        self.lineEdit_8.setEnabled(False)
        self.lineEdit_8.setText("7")

        # Figure size y
        self.lineEdit_9 = QtWidgets.QLineEdit(parent=self.tab_3)
        self.lineEdit_9.setGeometry(QtCore.QRect(550, 170, 113, 22))
        self.lineEdit_9.setObjectName("lineEdit_9")
        self.lineEdit_9.setEnabled(False)
        self.lineEdit_9.setText("6")

        # Figure size x
        self.label_11_1 = QtWidgets.QLabel(parent=self.tab_3)
        self.label_11_1.setGeometry(QtCore.QRect(450, 170, 81, 21))
        self.label_11_1.setObjectName("label_11_1")

        # Figure size y
        self.label_11_2 = QtWidgets.QLabel(parent=self.tab_3)
        self.label_11_2.setGeometry(QtCore.QRect(670, 170, 81, 21))
        self.label_11_2.setObjectName("label_11_2")

        # Pushbutton_4_generates the purple images in another tab
        self.binary_mask_nafion_DF = None
        self.binary_mask_cat_DF = None
        self.binary_mask_nafion = None
        self.binary_mask_cat = None
        self.pushButton_4 = QtWidgets.QPushButton(parent=self.tab_3)
        self.pushButton_4.setGeometry(QtCore.QRect(10, 10, 171, 24))
        self.pushButton_4.setObjectName("pushButton_4")
        self.pushButton_4.setEnabled(False)
        self.pushButton_4.clicked.connect(self.handle_binary_plots)

        # Huge display for Binary masks of catalyst and nafion
        self.graphicsView_6 = QtWidgets.QGraphicsView(parent=self.tab_3)
        self.graphicsView_6.setGeometry(QtCore.QRect(160, 200, 771, 471))
        self.graphicsView_6.setObjectName("graphicsView_6")

        self.tabWidget.addTab(self.tab_3, "")

        # endregion

        # Tab 4---------------------------------------------------------------------------------------------------------
        # region
        self.tab_4 = QtWidgets.QWidget()
        self.tab_4.setObjectName("tab_4")

        # Top view, before fine adjustment
        self.graphicsView_7 = QtWidgets.QGraphicsView(parent=self.tab_4)
        self.graphicsView_7.setGeometry(QtCore.QRect(190, 70, 891, 381))
        self.graphicsView_7.setObjectName("graphicsView_7")

        # Bottom view, after fine adjustment
        self.graphicsView_8 = QtWidgets.QGraphicsView(parent=self.tab_4)
        self.graphicsView_8.setGeometry(QtCore.QRect(190, 480, 891, 381))
        self.graphicsView_8.setObjectName("graphicsView_8")

        # Segment button

        self.selection1_DF = None
        self.selectionPt_DF = None
        self.selectionIr_DF = None
        self.gray_values_real_DF = None
        self.selectionPt = None
        self.selectionir = None
        self.real_im = None
        self.figure_parameters_segment = None
        self.pushButton_5 = QtWidgets.QPushButton(parent=self.tab_4)
        self.pushButton_5.setGeometry(QtCore.QRect(20, 30, 151, 24))
        self.pushButton_5.setObjectName("pushButton_5")
        self.pushButton_5.setEnabled(False)  # Initially disabled
        self.pushButton_5.clicked.connect(self.handle_segment)

        # Fine tune button
        self.contour_second_Ir = None
        self.biglist_nafion = None
        self.nafion_membrane_DF = None
        self.nafion_membrane_DF1 = None
        self.pushButton_6 = QtWidgets.QPushButton(parent=self.tab_4)
        self.pushButton_6.setGeometry(QtCore.QRect(20, 560, 151, 24))
        self.pushButton_6.setObjectName("pushButton_6")
        self.pushButton_6.setEnabled(False)  # Initially disabled
        self.pushButton_6.clicked.connect(self.handle_fine_tune)

        # Export coordinates to fiji button
        self.contour_second_Ir = None
        self.pushButton_7 = QtWidgets.QPushButton(parent=self.tab_4)
        self.pushButton_7.setGeometry(QtCore.QRect(20, 840, 151, 24))
        self.pushButton_7.setObjectName("pushButton_7")
        self.pushButton_7.setEnabled(False)  # Initially disabled
        self.pushButton_7.clicked.connect(self.handle_contour)

        # Fine tune input
        self.lineEdit_10 = QtWidgets.QLineEdit(parent=self.tab_4)
        self.lineEdit_10.setGeometry(QtCore.QRect(20, 520, 61, 22))
        self.lineEdit_10.setText("10")  # Set the initial text to "10 Px"
        self.lineEdit_10.setObjectName("lineEdit_10")
        self.lineEdit_10.setEnabled(False)  # Initially disabled

        # color_bar_shrink_plot1
        self.lineEdit_11 = QtWidgets.QLineEdit(parent=self.tab_4)
        self.lineEdit_11.setGeometry(QtCore.QRect(20, 150, 61, 22))
        self.lineEdit_11.setText("0.62")  # Set the initial text to "10 Px"
        self.lineEdit_11.setObjectName("lineEdit_11")
        self.lineEdit_11.setEnabled(False)  # Initially disabled

        # color_bar_shrink_plot2
        self.lineEdit_12 = QtWidgets.QLineEdit(parent=self.tab_4)
        self.lineEdit_12.setGeometry(QtCore.QRect(20, 220, 61, 22))
        self.lineEdit_12.setText("0.62")  # Set the initial text to "10 Px"
        self.lineEdit_12.setObjectName("lineEdit_12")
        self.lineEdit_12.setEnabled(False)  # Initially disabled

        # color_bar_shrink_plot3
        self.lineEdit_13 = QtWidgets.QLineEdit(parent=self.tab_4)
        self.lineEdit_13.setGeometry(QtCore.QRect(20, 280, 61, 22))
        self.lineEdit_13.setText("0.32")  # Set the initial text to "10 Px"
        self.lineEdit_13.setObjectName("lineEdit_13")
        self.lineEdit_13.setEnabled(False)  # Initially disabled

        # color_bar_shrink_plot4
        self.lineEdit_14 = QtWidgets.QLineEdit(parent=self.tab_4)
        self.lineEdit_14.setGeometry(QtCore.QRect(20, 340, 61, 22))
        self.lineEdit_14.setText("0.32")  # Set the initial text to "10 Px"
        self.lineEdit_14.setObjectName("lineEdit_14")
        self.lineEdit_14.setEnabled(False)  # Initially disabled

        # figure_parameter_segment_1 x axis
        self.lineEdit_15 = QtWidgets.QLineEdit(parent=self.tab_4)
        self.lineEdit_15.setGeometry(QtCore.QRect(420, 40, 113, 22))
        self.lineEdit_15.setText("10")  # Set the initial text to "10 Px"
        self.lineEdit_15.setObjectName("lineEdit_15")
        self.lineEdit_15.setEnabled(False)  # Initially disabled

        # figure_parameter_segment_2 y axis
        self.lineEdit_16 = QtWidgets.QLineEdit(parent=self.tab_4)
        self.lineEdit_16.setGeometry(QtCore.QRect(640, 40, 113, 22))
        self.lineEdit_16.setText("4")  # Set the initial text to "10 Px"
        self.lineEdit_16.setObjectName("lineEdit_16")
        self.lineEdit_16.setEnabled(False)  # Initially disabled

        # Fine tune label in pixels
        self.label_12 = QtWidgets.QLabel(parent=self.tab_4)
        self.label_12.setGeometry(QtCore.QRect(90, 520, 81, 16))
        self.label_12.setObjectName("label_12")

        # Color bar shrink (0-1) plot 1
        self.label_12_1 = QtWidgets.QLabel(parent=self.tab_4)
        self.label_12_1.setGeometry(QtCore.QRect(20, 120, 161, 31))
        self.label_12_1.setObjectName("label_12_1")

        # Color bar shrink (0-1) plot 2
        self.label_12_2 = QtWidgets.QLabel(parent=self.tab_4)
        self.label_12_2.setGeometry(QtCore.QRect(20, 190, 161, 31))
        self.label_12_2.setObjectName("label_12_2")

        # Color bar shrink (0-1) plot 3
        self.label_12_3 = QtWidgets.QLabel(parent=self.tab_4)
        self.label_12_3.setGeometry(QtCore.QRect(20, 250, 161, 31))
        self.label_12_3.setObjectName("label_12_3")

        # Color bar shrink (0-1) plot 4
        self.label_12_4 = QtWidgets.QLabel(parent=self.tab_4)
        self.label_12_4.setGeometry(QtCore.QRect(20, 310, 161, 31))
        self.label_12_4.setObjectName("label_12_4")

        # Figure size (x)
        self.label_12_5 = QtWidgets.QLabel(parent=self.tab_4)
        self.label_12_5.setGeometry(QtCore.QRect(540, 40, 81, 21))
        self.label_12_5.setObjectName("label_12_4")

        # Figure size (y)
        self.label_12_6 = QtWidgets.QLabel(parent=self.tab_4)
        self.label_12_6.setGeometry(QtCore.QRect(760, 40, 81, 21))
        self.label_12_6.setObjectName("label_12_6")

        # Proceed with fine tune checkbox
        self.checkBox = QtWidgets.QCheckBox(parent=self.tab_4)
        self.checkBox.setGeometry(QtCore.QRect(20, 480, 161, 20))
        self.checkBox.setObjectName("checkBox")
        self.checkBox.setText("Enable Fine Tune")
        self.checkBox.setEnabled(False)
        # Connect checkbox state to control widgets
        self.checkBox.stateChanged.connect(self.toggle_widgets)
        self.tabWidget.addTab(self.tab_4, "")

        # endregion

        # Tab 5---------------------------------------------------------------------------------------------------------
        # region
        self.tab_5 = QtWidgets.QWidget()
        self.tab_5.setObjectName("tab_5")

        # Graphics#
        # region
        # membrane 3D viewer
        self.graphicsView_9 = QtWidgets.QGraphicsView(parent=self.tab_5)
        self.graphicsView_9.setGeometry(QtCore.QRect(740, 10, 351, 301))
        self.graphicsView_9.setObjectName("graphicsView_9")
        # membrane histogram viewer
        self.graphicsView_10 = QtWidgets.QGraphicsView(parent=self.tab_5)
        self.graphicsView_10.setGeometry(QtCore.QRect(740, 320, 351, 301))
        self.graphicsView_10.setObjectName("graphicsView_10")
        # membrane otsu viewer
        self.graphicsView_11 = QtWidgets.QGraphicsView(parent=self.tab_5)
        self.graphicsView_11.setGeometry(QtCore.QRect(310, 320, 256, 301))
        self.graphicsView_11.setObjectName("graphicsView_11")
        # membrane viewer
        self.graphicsView_13 = QtWidgets.QGraphicsView(parent=self.tab_5)
        self.graphicsView_13.setGeometry(QtCore.QRect(310, 10, 256, 301))
        self.graphicsView_13.setObjectName("graphicsView_13")

        # main plot
        self.graphicsView_12 = QtWidgets.QGraphicsView(parent=self.tab_5)
        self.graphicsView_12.setGeometry(QtCore.QRect(190, 680, 901, 281))
        self.graphicsView_12.setObjectName("graphicsView_12")
        # endregion
        # ComboBox#
        # region

        self.comboBox_histogram_legend = QtWidgets.QComboBox(parent=self.tab_5)
        self.comboBox_histogram_legend.setGeometry(QtCore.QRect(20, 910, 91, 22))
        self.comboBox_histogram_legend.setObjectName("comboBox_histogram_legend")
        self.comboBox_histogram_legend.addItem("upper right")
        self.comboBox_histogram_legend.addItem("upper left")
        self.comboBox_histogram_legend.addItem("lower right")
        self.comboBox_histogram_legend.addItem("lower left")
        self.comboBox_histogram_legend.setEnabled(False)

        self.comboBox_membrane_legend = QtWidgets.QComboBox(parent=self.tab_5)
        self.comboBox_membrane_legend.setGeometry(QtCore.QRect(20, 940, 91, 22))
        self.comboBox_membrane_legend.setObjectName("comboBox_membrane_legend")
        self.comboBox_membrane_legend.addItem("upper right")
        self.comboBox_membrane_legend.addItem("upper left")
        self.comboBox_membrane_legend.addItem("lower right")
        self.comboBox_membrane_legend.addItem("lower left")
        self.comboBox_membrane_legend.setEnabled(False)

        self.comboBox_legend_position = QtWidgets.QComboBox(parent=self.tab_5)
        self.comboBox_legend_position.setGeometry(QtCore.QRect(600, 590, 91, 22))
        self.comboBox_legend_position.setObjectName("comboBox_legend_position")
        self.comboBox_legend_position.addItem("upper right")
        self.comboBox_legend_position.addItem("upper left")
        self.comboBox_legend_position.addItem("lower right")
        self.comboBox_legend_position.addItem("lower left")
        self.comboBox_legend_position.setEnabled(False)

        # endregion
        # Labels#
        # region
        self.label_13 = QtWidgets.QLabel(parent=self.tab_5)
        self.label_13.setGeometry(QtCore.QRect(40, 10, 101, 41))
        self.label_13.setObjectName("label_13")

        self.label_14 = QtWidgets.QLabel(parent=self.tab_5)
        self.label_14.setGeometry(QtCore.QRect(600, 10, 121, 16))
        self.label_14.setObjectName("label_14")

        self.label_15 = QtWidgets.QLabel(parent=self.tab_5)
        self.label_15.setGeometry(QtCore.QRect(600, 60, 121, 16))
        self.label_15.setObjectName("label_15")

        self.label_16 = QtWidgets.QLabel(parent=self.tab_5)
        self.label_16.setGeometry(QtCore.QRect(190, 470, 121, 16))
        self.label_16.setObjectName("label_16")

        self.label_17 = QtWidgets.QLabel(parent=self.tab_5)
        self.label_17.setGeometry(QtCore.QRect(190, 520, 121, 16))
        self.label_17.setObjectName("label_17")

        self.label_18 = QtWidgets.QLabel(parent=self.tab_5)
        self.label_18.setGeometry(QtCore.QRect(190, 330, 121, 16))
        self.label_18.setObjectName("label_18")

        self.label_19 = QtWidgets.QLabel(parent=self.tab_5)
        self.label_19.setGeometry(QtCore.QRect(190, 380, 121, 16))
        self.label_19.setObjectName("label_19")

        self.label_20 = QtWidgets.QLabel(parent=self.tab_5)
        self.label_20.setGeometry(QtCore.QRect(190, 570, 121, 16))
        self.label_20.setObjectName("label_20")

        self.label_21 = QtWidgets.QLabel(parent=self.tab_5)
        self.label_21.setGeometry(QtCore.QRect(600, 120, 131, 16))
        self.label_21.setObjectName("label_21")

        self.label_22 = QtWidgets.QLabel(parent=self.tab_5)
        self.label_22.setGeometry(QtCore.QRect(600, 170, 121, 16))
        self.label_22.setObjectName("label_22")

        self.label_23 = QtWidgets.QLabel(parent=self.tab_5)
        self.label_23.setGeometry(QtCore.QRect(650, 650, 121, 21))
        self.label_23.setObjectName("label_23")

        self.label_24 = QtWidgets.QLabel(parent=self.tab_5)
        self.label_24.setGeometry(QtCore.QRect(430, 650, 121, 21))
        self.label_24.setObjectName("label_24")

        self.label_25 = QtWidgets.QLabel(parent=self.tab_5)
        self.label_25.setGeometry(QtCore.QRect(40, 650, 121, 16))
        self.label_25.setObjectName("label_25")

        self.label_26 = QtWidgets.QLabel(parent=self.tab_5)
        self.label_26.setGeometry(QtCore.QRect(40, 700, 121, 16))
        self.label_26.setObjectName("label_26")

        self.label_27 = QtWidgets.QLabel(parent=self.tab_5)
        self.label_27.setGeometry(QtCore.QRect(40, 750, 121, 16))
        self.label_27.setObjectName("label_27")

        self.label_28 = QtWidgets.QLabel(parent=self.tab_5)
        self.label_28.setGeometry(QtCore.QRect(40, 850, 121, 16))
        self.label_28.setObjectName("label_28")

        self.label_29 = QtWidgets.QLabel(parent=self.tab_5)
        self.label_29.setGeometry(QtCore.QRect(40, 800, 121, 16))
        self.label_29.setObjectName("label_29")

        self.label_30 = QtWidgets.QLabel(parent=self.tab_5)
        self.label_30.setGeometry(QtCore.QRect(120, 910, 61, 20))
        self.label_30.setObjectName("label_30")

        self.label_31 = QtWidgets.QLabel(parent=self.tab_5)
        self.label_31.setGeometry(QtCore.QRect(120, 940, 71, 20))
        self.label_31.setObjectName("label_31")

        self.label_32 = QtWidgets.QLabel(parent=self.tab_5)
        self.label_32.setGeometry(QtCore.QRect(600, 470, 121, 16))
        self.label_32.setObjectName("label_32")

        self.label_33 = QtWidgets.QLabel(parent=self.tab_5)
        self.label_33.setGeometry(QtCore.QRect(600, 520, 121, 16))
        self.label_33.setObjectName("label_33")

        self.label_34 = QtWidgets.QLabel(parent=self.tab_5)
        self.label_34.setGeometry(QtCore.QRect(600, 380, 121, 16))
        self.label_34.setObjectName("label_34")

        self.label_35 = QtWidgets.QLabel(parent=self.tab_5)
        self.label_35.setGeometry(QtCore.QRect(600, 330, 121, 16))
        self.label_35.setObjectName("label_35")

        self.label_36 = QtWidgets.QLabel(parent=self.tab_5)
        self.label_36.setGeometry(QtCore.QRect(600, 570, 101, 16))
        self.label_36.setObjectName("label_36")

        # endregion
        # Line_edits#
        # region

        # Adding QLineEdit widgets with QDoubleValidator for both integers and floats

        # filter_for_dust
        self.lineEdit_17 = QtWidgets.QLineEdit(parent=self.tab_5)
        self.lineEdit_17.setGeometry(QtCore.QRect(30, 50, 113, 22))
        self.lineEdit_17.setObjectName("lineEdit_17")
        self.lineEdit_17.setValidator(QDoubleValidator())
        self.lineEdit_17.setEnabled(False)
        self.lineEdit_17.setText("15")

        # ------3D PLOT
        # set_tick_position_start
        self.lineEdit_18 = QtWidgets.QLineEdit(parent=self.tab_5)
        self.lineEdit_18.setGeometry(QtCore.QRect(600, 30, 113, 22))
        self.lineEdit_18.setObjectName("lineEdit_18")
        self.lineEdit_18.setValidator(QDoubleValidator())
        self.lineEdit_18.setEnabled(False)
        self.lineEdit_18.setText("0")

        # set_tick_position_end
        self.lineEdit_19 = QtWidgets.QLineEdit(parent=self.tab_5)
        self.lineEdit_19.setGeometry(QtCore.QRect(600, 80, 113, 22))
        self.lineEdit_19.setObjectName("lineEdit_19")
        self.lineEdit_19.setValidator(QDoubleValidator())
        self.lineEdit_19.setEnabled(False)
        self.lineEdit_19.setText("140")

        # elevation_view
        self.lineEdit_20 = QtWidgets.QLineEdit(parent=self.tab_5)
        self.lineEdit_20.setGeometry(QtCore.QRect(600, 190, 113, 22))
        self.lineEdit_20.setObjectName("lineEdit_20")
        self.lineEdit_20.setValidator(QDoubleValidator())
        self.lineEdit_20.setEnabled(False)
        self.lineEdit_20.setText("60")

        # azimutal_view
        self.lineEdit_21 = QtWidgets.QLineEdit(parent=self.tab_5)
        self.lineEdit_21.setGeometry(QtCore.QRect(600, 140, 113, 22))
        self.lineEdit_21.setObjectName("lineEdit_21")
        self.lineEdit_21.setValidator(QDoubleValidator())
        self.lineEdit_21.setEnabled(False)
        self.lineEdit_21.setText("60")


        # ------ Membrane - gray after Multi-otsu

        # color_bar_max_membrane_1
        self.lineEdit_22 = QtWidgets.QLineEdit(parent=self.tab_5)
        self.lineEdit_22.setGeometry(QtCore.QRect(190, 540, 113, 22))
        self.lineEdit_22.setObjectName("lineEdit_22")
        self.lineEdit_22.setValidator(QDoubleValidator())
        self.lineEdit_22.setEnabled(False)
        self.lineEdit_22.setText("15")

        # color_bar_min_membrane_1
        self.lineEdit_23 = QtWidgets.QLineEdit(parent=self.tab_5)
        self.lineEdit_23.setGeometry(QtCore.QRect(190, 490, 111, 22))
        self.lineEdit_23.setObjectName("lineEdit_23")
        self.lineEdit_23.setValidator(QDoubleValidator())
        self.lineEdit_23.setEnabled(False)
        self.lineEdit_23.setText("0")

        # figure_size_x_membrane_1
        self.lineEdit_24 = QtWidgets.QLineEdit(parent=self.tab_5)
        self.lineEdit_24.setGeometry(QtCore.QRect(190, 350, 113, 22))
        self.lineEdit_24.setObjectName("lineEdit_24")
        self.lineEdit_24.setValidator(QDoubleValidator())
        self.lineEdit_24.setEnabled(False)
        self.lineEdit_24.setText("5")

        # figure_size_y_membrane_1
        self.lineEdit_25 = QtWidgets.QLineEdit(parent=self.tab_5)
        self.lineEdit_25.setGeometry(QtCore.QRect(190, 400, 113, 22))
        self.lineEdit_25.setObjectName("lineEdit_25")
        self.lineEdit_25.setValidator(QDoubleValidator())
        self.lineEdit_25.setEnabled(False)
        self.lineEdit_25.setText("4")

        # color_bar_shrink_membrane_1
        self.lineEdit_26 = QtWidgets.QLineEdit(parent=self.tab_5)
        self.lineEdit_26.setGeometry(QtCore.QRect(190, 590, 113, 22))
        self.lineEdit_26.setObjectName("lineEdit_26")
        self.lineEdit_26.setValidator(QDoubleValidator())
        self.lineEdit_26.setEnabled(False)
        self.lineEdit_26.setText("1")


        # ------ Membrane histogram

        # histogram_limit_y_max_1
        self.lineEdit_27 = QtWidgets.QLineEdit(parent=self.tab_5)
        self.lineEdit_27.setGeometry(QtCore.QRect(600, 540, 102, 22))
        self.lineEdit_27.setObjectName("lineEdit_27")
        self.lineEdit_27.setValidator(QDoubleValidator())
        self.lineEdit_27.setEnabled(False)
        self.lineEdit_27.setText("1")

        # histogram_limit_x_max_1
        self.lineEdit_28 = QtWidgets.QLineEdit(parent=self.tab_5)
        self.lineEdit_28.setGeometry(QtCore.QRect(660, 490, 51, 22))
        self.lineEdit_28.setObjectName("lineEdit_28")
        self.lineEdit_28.setValidator(QDoubleValidator())
        self.lineEdit_28.setEnabled(False)
        self.lineEdit_28.setText("12")

        # histogram_limit_x_min_1
        self.lineEdit_29 = QtWidgets.QLineEdit(parent=self.tab_5)
        self.lineEdit_29.setGeometry(QtCore.QRect(600, 490, 51, 22))
        self.lineEdit_29.setObjectName("lineEdit_29")
        self.lineEdit_29.setValidator(QDoubleValidator())
        self.lineEdit_29.setEnabled(False)
        self.lineEdit_29.setText("0")

        # figure_size_x_histogram_1
        self.lineEdit_30 = QtWidgets.QLineEdit(parent=self.tab_5)
        self.lineEdit_30.setGeometry(QtCore.QRect(600, 400, 113, 22))
        self.lineEdit_30.setObjectName("lineEdit_30")
        self.lineEdit_30.setValidator(QDoubleValidator())
        self.lineEdit_30.setEnabled(False)
        self.lineEdit_30.setText("4")

        # figure_size_y_histogram_1
        self.lineEdit_31 = QtWidgets.QLineEdit(parent=self.tab_5)
        self.lineEdit_31.setGeometry(QtCore.QRect(600, 350, 113, 22))
        self.lineEdit_31.setObjectName("lineEdit_31")
        self.lineEdit_31.setValidator(QDoubleValidator())
        self.lineEdit_31.setEnabled(False)
        self.lineEdit_31.setText("4")

        # ------Main plot
        # figure_size_y_main_plot_1 32
        self.lineEdit_32 = QtWidgets.QLineEdit(parent=self.tab_5)
        self.lineEdit_32.setGeometry(QtCore.QRect(730, 650, 113, 22))
        self.lineEdit_32.setObjectName("lineEdit_32")
        self.lineEdit_32.setValidator(QDoubleValidator())
        self.lineEdit_32.setEnabled(False)
        self.lineEdit_32.setText("4")

        # figure_size_x_main_plot_1 33
        self.lineEdit_33 = QtWidgets.QLineEdit(parent=self.tab_5)
        self.lineEdit_33.setGeometry(QtCore.QRect(510, 650, 113, 22))
        self.lineEdit_33.setObjectName("lineEdit_33")
        self.lineEdit_33.setValidator(QDoubleValidator())
        self.lineEdit_33.setEnabled(False)
        self.lineEdit_33.setText("16")

        # color_bar_min_main_plot_1 34
        self.lineEdit_34 = QtWidgets.QLineEdit(parent=self.tab_5)
        self.lineEdit_34.setGeometry(QtCore.QRect(40, 670, 113, 22))
        self.lineEdit_34.setObjectName("lineEdit_34")
        self.lineEdit_34.setValidator(QDoubleValidator())
        self.lineEdit_34.setEnabled(False)
        self.lineEdit_34.setText("0")

        # color_bar_shrink_main_plot_1 35
        self.lineEdit_35 = QtWidgets.QLineEdit(parent=self.tab_5)
        self.lineEdit_35.setGeometry(QtCore.QRect(40, 770, 113, 22))
        self.lineEdit_35.setObjectName("lineEdit_35")
        self.lineEdit_35.setValidator(QDoubleValidator())
        self.lineEdit_35.setEnabled(False)
        self.lineEdit_35.setText("1")

        # color_bar_max_main_plot_1 36
        self.lineEdit_36 = QtWidgets.QLineEdit(parent=self.tab_5)
        self.lineEdit_36.setGeometry(QtCore.QRect(40, 720, 113, 22))
        self.lineEdit_36.setObjectName("lineEdit_36")
        self.lineEdit_36.setValidator(QDoubleValidator())
        self.lineEdit_36.setEnabled(False)
        self.lineEdit_36.setText("15")

        # main_plot_histogram_limit_x_min_1 37
        self.lineEdit_37 = QtWidgets.QLineEdit(parent=self.tab_5)
        self.lineEdit_37.setGeometry(QtCore.QRect(40, 820, 51, 22))
        self.lineEdit_37.setObjectName("lineEdit_37")
        self.lineEdit_37.setValidator(QDoubleValidator())
        self.lineEdit_37.setEnabled(False)
        self.lineEdit_37.setText("0")

        # main_plot_histogram_limit_x_max_1 38
        self.lineEdit_38 = QtWidgets.QLineEdit(parent=self.tab_5)
        self.lineEdit_38.setGeometry(QtCore.QRect(100, 820, 51, 22))
        self.lineEdit_38.setObjectName("lineEdit_38")
        self.lineEdit_38.setValidator(QDoubleValidator())
        self.lineEdit_38.setEnabled(False)
        self.lineEdit_38.setText("12")

        # main_plot_histogram_limit_y_max_1 39
        self.lineEdit_39 = QtWidgets.QLineEdit(parent=self.tab_5)
        self.lineEdit_39.setGeometry(QtCore.QRect(40, 870, 113, 22))
        self.lineEdit_39.setObjectName("lineEdit_39")
        self.lineEdit_39.setValidator(QDoubleValidator())
        self.lineEdit_39.setEnabled(False)
        self.lineEdit_39.setText("1.2")

        # ------
        # endregion
        # Lines#
        # region

        self.line = QtWidgets.QFrame(parent=self.tab_5)
        self.line.setGeometry(QtCore.QRect(0, 630, 1111, 16))
        self.line.setFrameShape(QtWidgets.QFrame.Shape.HLine)
        self.line.setFrameShadow(QtWidgets.QFrame.Shadow.Sunken)
        self.line.setObjectName("line")

        self.line_2 = QtWidgets.QFrame(parent=self.tab_5)
        self.line_2.setGeometry(QtCore.QRect(170, 10, 20, 621))
        self.line_2.setFrameShape(QtWidgets.QFrame.Shape.VLine)
        self.line_2.setFrameShadow(QtWidgets.QFrame.Shadow.Sunken)
        self.line_2.setObjectName("line_2")

        # endregion
        # Pushbuttons#
        # region

        # apply filter button
        self.nafion_membrane_DF2 = None
        self.pushButton_8 = QtWidgets.QPushButton(parent=self.tab_5)
        self.pushButton_8.setGeometry(QtCore.QRect(50, 80, 75, 24))
        self.pushButton_8.setObjectName("pushButton_8")
        self.pushButton_8.setEnabled(False)
        self.pushButton_8.clicked.connect(self.handle_filter_dust)

        # segment nafion membrane
        self.images = None
        self.pushButton_9 = QtWidgets.QPushButton(parent=self.tab_5)
        self.pushButton_9.setGeometry(QtCore.QRect(190, 650, 131, 24))
        self.pushButton_9.setStyleSheet("")
        self.pushButton_9.setObjectName("pushButton_9")
        self.pushButton_9.setEnabled(False)
        self.pushButton_9.clicked.connect(self.handle_segment_nafion)

        # load pristine
        self.address_csv1 = None  # bins for pristine - gray values normalized
        self.address_csv2 = None  # counts for pristine - gray values normalized
        self.address_csv3 = None  # bins for pristine - quantified normalized
        self.address_csv4 = None  # counts for pristine - quantified normalized
        self.pushButton_load_pristine = QtWidgets.QPushButton(parent=self.tab_5)
        self.pushButton_load_pristine.setGeometry(QtCore.QRect(950, 650, 138, 24))
        self.pushButton_load_pristine.setStyleSheet("")
        self.pushButton_load_pristine.setObjectName("pushButton_9")
        self.pushButton_load_pristine.setEnabled(False)
        self.pushButton_load_pristine.clicked.connect(self.load_pristine)
        # endregion

        self.tabWidget.addTab(self.tab_5, "")

        # endregion

        # Tab 6---------------------------------------------------------------------------------------------------------
        #region
        self.tab_6 = QtWidgets.QWidget()
        self.tab_6.setObjectName("tab_6")

        # Graphics#
        #region
        self.graphicsView_14 = QtWidgets.QGraphicsView(parent=self.tab_6)
        self.graphicsView_14.setGeometry(QtCore.QRect(170, 80, 911, 801))
        self.graphicsView_14.setObjectName("graphicsView_14")
        #endregion
        # Label#
        #region
        self.label_37 = QtWidgets.QLabel(parent=self.tab_6)
        self.label_37.setGeometry(QtCore.QRect(30, 215, 121, 21))
        self.label_37.setObjectName("label_37")

        self.label_38 = QtWidgets.QLabel(parent=self.tab_6)
        self.label_38.setGeometry(QtCore.QRect(30, 115, 121, 21))
        self.label_38.setObjectName("label_38")

        self.label_39 = QtWidgets.QLabel(parent=self.tab_6)
        self.label_39.setGeometry(QtCore.QRect(30, 165, 121, 21))
        self.label_39.setObjectName("label_39")

        self.label_40 = QtWidgets.QLabel(parent=self.tab_6)
        self.label_40.setGeometry(QtCore.QRect(30, 365, 121, 21))
        self.label_40.setObjectName("label_40")

        self.label_41 = QtWidgets.QLabel(parent=self.tab_6)
        self.label_41.setGeometry(QtCore.QRect(30, 315, 121, 21))
        self.label_41.setObjectName("label_41")

        self.label_42 = QtWidgets.QLabel(parent=self.tab_6)
        self.label_42.setGeometry(QtCore.QRect(30, 415, 121, 21))
        self.label_42.setObjectName("label_42")

        self.label_43 = QtWidgets.QLabel(parent=self.tab_6)
        self.label_43.setGeometry(QtCore.QRect(40, 90, 111, 21))
        self.label_43.setObjectName("label_43")

        self.label_44 = QtWidgets.QLabel(parent=self.tab_6)
        self.label_44.setGeometry(QtCore.QRect(290, 30, 191, 21))
        self.label_44.setObjectName("label_44")

        self.label_45 = QtWidgets.QLabel(parent=self.tab_6)
        self.label_45.setGeometry(QtCore.QRect(40, 290, 111, 21))
        self.label_45.setObjectName("label_45")

        self.label_46 = QtWidgets.QLabel(parent=self.tab_6)
        self.label_46.setGeometry(QtCore.QRect(860, 1080, 111, 21))
        self.label_46.setObjectName("label_46")

        self.label_47 = QtWidgets.QLabel(parent=self.tab_6)
        self.label_47.setGeometry(QtCore.QRect(30, 565, 121, 21))
        self.label_47.setObjectName("label_47")

        self.label_48 = QtWidgets.QLabel(parent=self.tab_6)
        self.label_48.setGeometry(QtCore.QRect(30, 515, 121, 21))
        self.label_48.setObjectName("label_48")

        self.label_49 = QtWidgets.QLabel(parent=self.tab_6)
        self.label_49.setGeometry(QtCore.QRect(30, 615, 101, 21))
        self.label_49.setObjectName("label_49")

        self.label_50 = QtWidgets.QLabel(parent=self.tab_6)
        self.label_50.setGeometry(QtCore.QRect(40, 690, 111, 21))
        self.label_50.setObjectName("label_50")

        self.label_51 = QtWidgets.QLabel(parent=self.tab_6)
        self.label_51.setGeometry(QtCore.QRect(30, 715, 121, 21))
        self.label_51.setObjectName("label_51")

        self.label_52 = QtWidgets.QLabel(parent=self.tab_6)
        self.label_52.setGeometry(QtCore.QRect(30, 765, 121, 21))
        self.label_52.setObjectName("label_52")

        self.label_53 = QtWidgets.QLabel(parent=self.tab_6)
        self.label_53.setGeometry(QtCore.QRect(30, 815, 101, 21))
        self.label_53.setObjectName("label_53")

        self.label_54 = QtWidgets.QLabel(parent=self.tab_6)
        self.label_54.setGeometry(QtCore.QRect(620, 30, 201, 21))
        self.label_54.setObjectName("label_54")

        self.label_55 = QtWidgets.QLabel(parent=self.tab_6)
        self.label_55.setGeometry(QtCore.QRect(890, 10, 81, 21))
        self.label_55.setObjectName("label_55")

        self.label_56 = QtWidgets.QLabel(parent=self.tab_6)
        self.label_56.setGeometry(QtCore.QRect(890, 50, 81, 21))
        self.label_56.setObjectName("label_56")

        self.label_57 = QtWidgets.QLabel(parent=self.tab_6)
        self.label_57.setGeometry(QtCore.QRect(30, 490, 111, 21))
        self.label_57.setObjectName("label_57")

        #endregion
        # LineEdit#
        #region

        #Gray scale values normal plot---------------------------------
        # color_bar_min_gray
        self.lineEdit_40 = QtWidgets.QLineEdit(parent=self.tab_6)
        self.lineEdit_40.setGeometry(QtCore.QRect(30, 140, 113, 22))
        self.lineEdit_40.setObjectName("lineEdit_40")
        self.lineEdit_40.setText("0.265")

        # color_bar_shrink_gray
        self.lineEdit_41 = QtWidgets.QLineEdit(parent=self.tab_6)
        self.lineEdit_41.setGeometry(QtCore.QRect(30, 240, 113, 22))
        self.lineEdit_41.setObjectName("lineEdit_41")
        self.lineEdit_41.setText("1")

        # color_bar_max_gray
        self.lineEdit_42 = QtWidgets.QLineEdit(parent=self.tab_6)
        self.lineEdit_42.setGeometry(QtCore.QRect(30, 190, 113, 22))
        self.lineEdit_42.setObjectName("lineEdit_42")
        self.lineEdit_42.setText("0.280")

        # Thickness values normal plot---------------------------------

        # color_bar_min_thickness
        self.lineEdit_43 = QtWidgets.QLineEdit(parent=self.tab_6)
        self.lineEdit_43.setGeometry(QtCore.QRect(30, 340, 113, 22))
        self.lineEdit_43.setObjectName("lineEdit_43")
        self.lineEdit_43.setText("0")

        # color_bar_max_thickness
        self.lineEdit_44 = QtWidgets.QLineEdit(parent=self.tab_6)
        self.lineEdit_44.setGeometry(QtCore.QRect(30, 390, 113, 22))
        self.lineEdit_44.setObjectName("lineEdit_44")
        self.lineEdit_44.setText("45")

        # color_bar_shrink_thickness
        self.lineEdit_45 = QtWidgets.QLineEdit(parent=self.tab_6)
        self.lineEdit_45.setGeometry(QtCore.QRect(30, 440, 113, 22))
        self.lineEdit_45.setObjectName("lineEdit_45")
        self.lineEdit_45.setText("1")

        # Gray scale values histogram plot---------------------------------

        # histogram_y_max_gray
        self.lineEdit_46 = QtWidgets.QLineEdit(parent=self.tab_6)
        self.lineEdit_46.setGeometry(QtCore.QRect(30, 590, 102, 22))
        self.lineEdit_46.setObjectName("lineEdit_46")
        self.lineEdit_46.setText("1.2")

        # histogram_x_max_gray
        self.lineEdit_47 = QtWidgets.QLineEdit(parent=self.tab_6)
        self.lineEdit_47.setGeometry(QtCore.QRect(90, 540, 51, 22))
        self.lineEdit_47.setObjectName("lineEdit_47")
        self.lineEdit_47.setText("0.280")

        # histogram_x_min_gray
        self.lineEdit_48 = QtWidgets.QLineEdit(parent=self.tab_6)
        self.lineEdit_48.setGeometry(QtCore.QRect(30, 540, 51, 22))
        self.lineEdit_48.setObjectName("lineEdit_48")
        self.lineEdit_48.setText("0.265")

        # Thickness values histogram plot---------------------------------

        # histogram_x_max_thickness
        self.lineEdit_49 = QtWidgets.QLineEdit(parent=self.tab_6)
        self.lineEdit_49.setGeometry(QtCore.QRect(90, 740, 51, 22))
        self.lineEdit_49.setObjectName("lineEdit_49")
        self.lineEdit_49.setText("44")

        # histogram_y_max_thickness
        self.lineEdit_50 = QtWidgets.QLineEdit(parent=self.tab_6)
        self.lineEdit_50.setGeometry(QtCore.QRect(30, 790, 102, 22))
        self.lineEdit_50.setObjectName("lineEdit_50")
        self.lineEdit_50.setText("1.2")

        # histogram_x_min_thickness
        self.lineEdit_51 = QtWidgets.QLineEdit(parent=self.tab_6)
        self.lineEdit_51.setGeometry(QtCore.QRect(30, 740, 51, 22))
        self.lineEdit_51.setObjectName("lineEdit_51")
        self.lineEdit_51.setText("29")

        # Main plot parameters---------------------------------

        # air_absorption_mean
        self.lineEdit_52 = QtWidgets.QLineEdit(parent=self.tab_6)
        self.lineEdit_52.setGeometry(QtCore.QRect(170, 30, 113, 22))
        self.lineEdit_52.setObjectName("lineEdit_52")
        self.lineEdit_52.setText("0.2331723")

        # attenuation_length
        self.lineEdit_53 = QtWidgets.QLineEdit(parent=self.tab_6)
        self.lineEdit_53.setGeometry(QtCore.QRect(500, 30, 113, 22))
        self.lineEdit_53.setObjectName("lineEdit_53")
        self.lineEdit_53.setText("926")

        # figure_size_x
        self.lineEdit_54 = QtWidgets.QLineEdit(parent=self.tab_6)
        self.lineEdit_54.setGeometry(QtCore.QRect(970, 10, 31, 22))
        self.lineEdit_54.setObjectName("lineEdit_54")
        self.lineEdit_54.setText("8")

        # figure_size_y
        self.lineEdit_55 = QtWidgets.QLineEdit(parent=self.tab_6)
        self.lineEdit_55.setGeometry(QtCore.QRect(970, 50, 31, 22))
        self.lineEdit_55.setObjectName("lineEdit_55")
        self.lineEdit_55.setText("8")

        # List of all QLineEdit objects
        self.line_edits = [
            self.lineEdit_40, self.lineEdit_41, self.lineEdit_42,
            self.lineEdit_43, self.lineEdit_44, self.lineEdit_45,
            self.lineEdit_46, self.lineEdit_47, self.lineEdit_48,
            self.lineEdit_49, self.lineEdit_50, self.lineEdit_51,
            self.lineEdit_52, self.lineEdit_53, self.lineEdit_54,
            self.lineEdit_55
        ]

        # Disable all line edits
        for line_edit in self.line_edits:
            line_edit.setEnabled(False)

        #endregion
        # Lines#
        #region
        self.line_3 = QtWidgets.QFrame(parent=self.tab_6)
        self.line_3.setGeometry(QtCore.QRect(10, 270, 151, 20))
        self.line_3.setFrameShape(QtWidgets.QFrame.Shape.HLine)
        self.line_3.setFrameShadow(QtWidgets.QFrame.Shadow.Sunken)
        self.line_3.setObjectName("line_3")
        self.line_5 = QtWidgets.QFrame(parent=self.tab_6)
        self.line_5.setGeometry(QtCore.QRect(10, 70, 151, 20))
        self.line_5.setFrameShape(QtWidgets.QFrame.Shape.HLine)
        self.line_5.setFrameShadow(QtWidgets.QFrame.Shadow.Sunken)
        self.line_5.setObjectName("line_5")
        self.line_6 = QtWidgets.QFrame(parent=self.tab_6)
        self.line_6.setGeometry(QtCore.QRect(10, 470, 151, 20))
        self.line_6.setFrameShape(QtWidgets.QFrame.Shape.HLine)
        self.line_6.setFrameShadow(QtWidgets.QFrame.Shadow.Sunken)
        self.line_6.setObjectName("line_6")
        self.line_14 = QtWidgets.QFrame(parent=self.tab_6)
        self.line_14.setGeometry(QtCore.QRect(10, 670, 151, 20))
        self.line_14.setFrameShape(QtWidgets.QFrame.Shape.HLine)
        self.line_14.setFrameShadow(QtWidgets.QFrame.Shadow.Sunken)
        self.line_14.setObjectName("line_14")
        self.line_15 = QtWidgets.QFrame(parent=self.tab_6)
        self.line_15.setGeometry(QtCore.QRect(310, 1080, 151, 20))
        self.line_15.setFrameShape(QtWidgets.QFrame.Shape.HLine)
        self.line_15.setFrameShadow(QtWidgets.QFrame.Shadow.Sunken)
        self.line_15.setObjectName("line_15")
        self.line_16 = QtWidgets.QFrame(parent=self.tab_6)
        self.line_16.setGeometry(QtCore.QRect(10, 870, 151, 20))
        self.line_16.setFrameShape(QtWidgets.QFrame.Shape.HLine)
        self.line_16.setFrameShadow(QtWidgets.QFrame.Shadow.Sunken)
        self.line_16.setObjectName("line_16")
        #endregion
        # ComboBox#
        #region
        self.comboBox_12 = QtWidgets.QComboBox(parent=self.tab_6)
        self.comboBox_12.setGeometry(QtCore.QRect(30, 640, 91, 22))
        self.comboBox_12.setObjectName("comboBox_12")
        self.comboBox_12.addItem("upper right")
        self.comboBox_12.addItem("upper left")
        self.comboBox_12.addItem("lower right")
        self.comboBox_12.addItem("lower left")
        self.comboBox_12.setEnabled(False)

        self.comboBox_13 = QtWidgets.QComboBox(parent=self.tab_6)
        self.comboBox_13.setGeometry(QtCore.QRect(30, 840, 91, 22))
        self.comboBox_13.setObjectName("comboBox_13")
        self.comboBox_13.addItem("upper right")
        self.comboBox_13.addItem("upper left")
        self.comboBox_13.addItem("lower right")
        self.comboBox_13.addItem("lower left")
        self.comboBox_13.setEnabled(False)

        #endregion
        # PushButton#
        #region

        self.gray_values_thickness_DF = None
        self.pushButton_10 = QtWidgets.QPushButton(parent=self.tab_6)
        self.pushButton_10.setGeometry(QtCore.QRect(30, 30, 111, 24))
        self.pushButton_10.setObjectName("pushButton_10")
        self.pushButton_10.clicked.connect(self.handle_thickness)
        self.pushButton_10.setEnabled(False)

        #endregion

        self.tabWidget.addTab(self.tab_6, "")

        #endregion

        # Tab 7---------------------------------------------------------------------------------------------------------
        #region
        self.tab_7 = QtWidgets.QWidget()
        self.tab_7.setObjectName("tab_7")

        # QGraphicsView
        # region
        self.graphicsView_15 = QtWidgets.QGraphicsView(parent=self.tab_7)
        self.graphicsView_15.setGeometry(QtCore.QRect(60, 350, 991, 561))
        self.graphicsView_15.setObjectName("graphicsView_14")

        self.graphicsView_16 = QtWidgets.QGraphicsView(parent=self.tab_7)
        self.graphicsView_16.setGeometry(QtCore.QRect(280, 930, 541, 31))
        self.graphicsView_16.setObjectName("graphicsView_15")
        # endregion QGraphicsView
        # QLineEdit
        # region
        self.lineEdit_56 = QtWidgets.QLineEdit(parent=self.tab_7)
        self.lineEdit_56.setGeometry(QtCore.QRect(270, 100, 111, 22))
        self.lineEdit_56.setObjectName("lineEdit_56")
        self.lineEdit_56.setText("4")  # y

        self.lineEdit_57 = QtWidgets.QLineEdit(parent=self.tab_7)
        self.lineEdit_57.setGeometry(QtCore.QRect(60, 100, 111, 22))
        self.lineEdit_57.setObjectName("lineEdit_57")
        self.lineEdit_57.setText("17")  # x

        self.lineEdit_58 = QtWidgets.QLineEdit(parent=self.tab_7)
        self.lineEdit_58.setGeometry(QtCore.QRect(430, 260, 113, 22))
        self.lineEdit_58.setObjectName("lineEdit_58")
        self.lineEdit_58.setText("0")  # start

        self.lineEdit_59 = QtWidgets.QLineEdit(parent=self.tab_7)
        self.lineEdit_59.setGeometry(QtCore.QRect(430, 210, 113, 22))
        self.lineEdit_59.setObjectName("lineEdit_59")
        self.lineEdit_59.setText("140")  # Set tick position: end

        self.lineEdit_60 = QtWidgets.QLineEdit(parent=self.tab_7)
        self.lineEdit_60.setGeometry(QtCore.QRect(570, 210, 113, 22))
        self.lineEdit_60.setObjectName("lineEdit_60")
        self.lineEdit_60.setText("60")  # elevation view

        self.lineEdit_61 = QtWidgets.QLineEdit(parent=self.tab_7)
        self.lineEdit_61.setGeometry(QtCore.QRect(570, 260, 113, 22))
        self.lineEdit_61.setObjectName("lineEdit_61")
        self.lineEdit_61.setText("60")  # azimutal view

        self.lineEdit_62 = QtWidgets.QLineEdit(parent=self.tab_7)
        self.lineEdit_62.setGeometry(QtCore.QRect(430, 310, 113, 22))
        self.lineEdit_62.setObjectName("lineEdit_62")
        self.lineEdit_62.setText("4")  # intrinsic axis shift for adjust

        self.lineEdit_63 = QtWidgets.QLineEdit(parent=self.tab_7)
        self.lineEdit_63.setGeometry(QtCore.QRect(770, 260, 111, 22))
        self.lineEdit_63.setObjectName("lineEdit_63")
        self.lineEdit_63.setText("1.2")  # Histogram limit (y)

        self.lineEdit_64 = QtWidgets.QLineEdit(parent=self.tab_7)
        self.lineEdit_64.setGeometry(QtCore.QRect(830, 210, 51, 22))
        self.lineEdit_64.setObjectName("lineEdit_64")
        self.lineEdit_64.setText("2.5e18")  # Histogram limit max (x)

        self.lineEdit_65 = QtWidgets.QLineEdit(parent=self.tab_7)
        self.lineEdit_65.setGeometry(QtCore.QRect(770, 210, 51, 22))
        self.lineEdit_65.setObjectName("lineEdit_65")
        self.lineEdit_65.setText("0.01e18")  # Histogram limit min (x)

        self.lineEdit_66 = QtWidgets.QLineEdit(parent=self.tab_7)
        self.lineEdit_66.setGeometry(QtCore.QRect(480, 100, 113, 22))
        self.lineEdit_66.setObjectName("lineEdit_66")
        self.lineEdit_66.setText("3.75E14")  # conversion factor

        self.lineEdit_67 = QtWidgets.QLineEdit(parent=self.tab_7)
        self.lineEdit_67.setGeometry(QtCore.QRect(100, 210, 113, 22))
        self.lineEdit_67.setObjectName("lineEdit_67")
        self.lineEdit_67.setText("0.1E18")  # Color bar min (a.u.)

        self.lineEdit_68 = QtWidgets.QLineEdit(parent=self.tab_7)
        self.lineEdit_68.setGeometry(QtCore.QRect(230, 210, 113, 22))
        self.lineEdit_68.setObjectName("lineEdit_68")
        self.lineEdit_68.setText("1")  # Color bar shrink (0-1)

        self.lineEdit_69 = QtWidgets.QLineEdit(parent=self.tab_7)
        self.lineEdit_69.setGeometry(QtCore.QRect(100, 260, 113, 22))
        self.lineEdit_69.setObjectName("lineEdit_69")
        self.lineEdit_69.setText("1E18")  # Color bar max (a.u.)

        # Updated list of QLineEdit objects
        self.line_edits_tab7 = [
            self.lineEdit_56, self.lineEdit_57, self.lineEdit_58, self.lineEdit_59,
            self.lineEdit_60, self.lineEdit_61, self.lineEdit_62, self.lineEdit_63,
            self.lineEdit_64, self.lineEdit_65, self.lineEdit_66, self.lineEdit_67,
            self.lineEdit_68, self.lineEdit_69
        ]

        # Disable all line edits
        for line_edit in self.line_edits_tab7:
            line_edit.setEnabled(False)

        # endregion
        # QLabel
        # region
        self.label_58 = QtWidgets.QLabel(parent=self.tab_7)
        self.label_58.setGeometry(QtCore.QRect(390, 100, 81, 21))
        self.label_58.setObjectName("label_58")

        self.label_59 = QtWidgets.QLabel(parent=self.tab_7)
        self.label_59.setGeometry(QtCore.QRect(180, 100, 81, 21))
        self.label_59.setObjectName("label_59")

        self.label_60 = QtWidgets.QLabel(parent=self.tab_7)
        self.label_60.setGeometry(QtCore.QRect(430, 190, 121, 16))
        self.label_60.setObjectName("label_60")

        self.label_61 = QtWidgets.QLabel(parent=self.tab_7)
        self.label_61.setGeometry(QtCore.QRect(570, 240, 121, 16))
        self.label_61.setObjectName("label_61")

        self.label_62 = QtWidgets.QLabel(parent=self.tab_7)
        self.label_62.setGeometry(QtCore.QRect(570, 190, 131, 16))
        self.label_62.setObjectName("label_62")

        self.label_63 = QtWidgets.QLabel(parent=self.tab_7)
        self.label_63.setGeometry(QtCore.QRect(430, 240, 121, 16))
        self.label_63.setObjectName("label_63")

        self.label_63_1 = QtWidgets.QLabel(parent=self.tab_7)
        self.label_63_1.setGeometry(QtCore.QRect(430, 290, 141, 21))
        self.label_63_1.setObjectName("label_63_1")

        self.label_64 = QtWidgets.QLabel(parent=self.tab_7)
        self.label_64.setGeometry(QtCore.QRect(770, 240, 121, 16))
        self.label_64.setObjectName("label_64")

        self.label_65 = QtWidgets.QLabel(parent=self.tab_7)
        self.label_65.setGeometry(QtCore.QRect(770, 190, 121, 16))
        self.label_65.setObjectName("label_65")

        self.label_66 = QtWidgets.QLabel(parent=self.tab_7)
        self.label_66.setGeometry(QtCore.QRect(910, 190, 101, 16))
        self.label_66.setObjectName("label_66")

        self.label_67 = QtWidgets.QLabel(parent=self.tab_7)
        self.label_67.setGeometry(QtCore.QRect(600, 100, 101, 21))
        self.label_67.setObjectName("label_67")

        self.label_68 = QtWidgets.QLabel(parent=self.tab_7)
        self.label_68.setGeometry(QtCore.QRect(230, 190, 121, 16))
        self.label_68.setObjectName("label_68")

        self.label_69 = QtWidgets.QLabel(parent=self.tab_7)
        self.label_69.setGeometry(QtCore.QRect(100, 190, 121, 16))
        self.label_69.setObjectName("label_69")

        self.label_70 = QtWidgets.QLabel(parent=self.tab_7)
        self.label_70.setGeometry(QtCore.QRect(100, 240, 121, 16))
        self.label_70.setObjectName("label_70")

        self.label_71 = QtWidgets.QLabel(parent=self.tab_7)
        self.label_71.setGeometry(QtCore.QRect(180, 160, 91, 16))
        self.label_71.setObjectName("label_71")

        self.label_72 = QtWidgets.QLabel(parent=self.tab_7)
        self.label_72.setGeometry(QtCore.QRect(860, 160, 61, 16))
        self.label_72.setObjectName("label_72")

        self.label_73 = QtWidgets.QLabel(parent=self.tab_7)
        self.label_73.setGeometry(QtCore.QRect(540, 160, 41, 16))
        self.label_73.setObjectName("label_73")
        # endregion QLabel
        # QComboBox
        # region
        self.comboBox_14 = QtWidgets.QComboBox(parent=self.tab_7)
        self.comboBox_14.setGeometry(QtCore.QRect(910, 210, 91, 22))
        self.comboBox_14.setObjectName("comboBox_14")
        self.comboBox_14.addItem("upper right")
        self.comboBox_14.addItem("upper left")
        self.comboBox_14.addItem("lower right")
        self.comboBox_14.addItem("lower left")
        self.comboBox_14.setEnabled(False)
        # endregion QComboBox
        # QFrame
        # region
        self.line_17 = QtWidgets.QFrame(parent=self.tab_7)
        self.line_17.setGeometry(QtCore.QRect(49, 190, 31, 141))
        self.line_17.setFrameShape(QtWidgets.QFrame.Shape.VLine)
        self.line_17.setFrameShadow(QtWidgets.QFrame.Shadow.Sunken)
        self.line_17.setObjectName("line_17")
        self.line_18 = QtWidgets.QFrame(parent=self.tab_7)
        self.line_18.setGeometry(QtCore.QRect(360, 190, 31, 141))
        self.line_18.setFrameShape(QtWidgets.QFrame.Shape.VLine)
        self.line_18.setFrameShadow(QtWidgets.QFrame.Shadow.Sunken)
        self.line_18.setObjectName("line_18")
        self.line_19 = QtWidgets.QFrame(parent=self.tab_7)
        self.line_19.setGeometry(QtCore.QRect(720, 190, 31, 141))
        self.line_19.setFrameShape(QtWidgets.QFrame.Shape.VLine)
        self.line_19.setFrameShadow(QtWidgets.QFrame.Shadow.Sunken)
        self.line_19.setObjectName("line_19")
        self.line_20 = QtWidgets.QFrame(parent=self.tab_7)
        self.line_20.setGeometry(QtCore.QRect(1030, 190, 31, 141))
        self.line_20.setFrameShape(QtWidgets.QFrame.Shape.VLine)
        self.line_20.setFrameShadow(QtWidgets.QFrame.Shadow.Sunken)
        self.line_20.setObjectName("line_20")

        #endregion
        # PushButton
        #region
        self.pushButton_11 = QtWidgets.QPushButton(parent=self.tab_7)
        self.pushButton_11.setGeometry(QtCore.QRect(60, 40, 161, 24))
        self.pushButton_11.setObjectName("pushButton_11")
        self.pushButton_11.setEnabled(False)
        self.pushButton_11.clicked.connect(self.handle_quantify)
        #endregion

        self.tabWidget.addTab(self.tab_7, "")

        #endregion


        ## Main window parameters ##

        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(parent=MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 1154, 22))
        self.menubar.setObjectName("menubar")
        MainWindow.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(parent=MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)

        self.retranslateUi(MainWindow)
        self.tabWidget.setCurrentIndex(0)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def handle_clear_all_graphics_views(self):
        """
        Clears all QGraphicsView instances in the given parent objects.
        :param parents: The parent objects containing the QGraphicsView instances.
        """
        for attr_name in dir(self):
            attr = getattr(self, attr_name)
            if isinstance(attr, QtWidgets.QGraphicsView):
                attr.setScene(None)
        print(f"Cleaning graphics")  # Optional debugging line

    def handle_start_project(self):
        try:

            # Prompt for folder selection -- storing future project
            QMessageBox.information(
                self.centralwidget,
                "Select Folder",
                "Please select a folder where data will be stored or processed later."
            )
            folder_path = QFileDialog.getExistingDirectory(
                self.centralwidget, "Select Folder",
                "D:/Data1/pshell_data"
            )

            if not folder_path:
                raise Exception("Folder not selected.")

            print(f"Selected folder: {folder_path}")


            # Prompt for the first TIFF stack file
            QMessageBox.information(
                self.centralwidget,
                "Select File",
                "Please select the first stack of images from sum of detectors (TIFF format)."
            )
            fname1, _ = QFileDialog.getOpenFileName(
                self.centralwidget, 'Open Stack of Images (TIFF) 1',
                'D:/x',
                'TIFF Files (*.tiff *.tif)'
            )

            if not fname1:
                raise Exception("Stack of images file not selected.")

            file_name1 = os.path.basename(fname1)
            print(f"Selected stack of images from sum of detectors (TIFF): {file_name1}")

            # Prompt for the second TIFF stack file
            QMessageBox.information(
                self.centralwidget,
                "Select File",
                "Please select the second stack of images from single detector (TIFF format)."
            )
            fname2, _ = QFileDialog.getOpenFileName(
                self.centralwidget, 'Open Stack of Images from single detector (TIFF)',
                'D:/x',
                'TIFF Files (*.tiff *.tif)'
            )

            if not fname2:
                raise Exception("Second stack of images file not selected.")

            file_name2 = os.path.basename(fname2)
            print(f"Selected stack of images from single detector (TIFF): {file_name2}")

            # Prompt for the third single TIFF file
            QMessageBox.information(
                self.centralwidget,
                "Select File",
                "Please select the single TIFF file."
            )
            fname3, _ = QFileDialog.getOpenFileName(
                self.centralwidget, 'Open Single TIFF File - mux_roi for absorption calculations',
                'D:/x',
                'TIFF Files (*.tiff *.tif)'
            )

            if not fname3:
                raise Exception("Single TIFF - mux_roi file not selected.")

            file_name3 = os.path.basename(fname3)
            print(f"Selected single TIFF - mux_roi file: {file_name3}")


            # Store addresses into variables
            self.project_path =  folder_path # folder to save project
            self.address_stack_tiff1 = fname1 # Stack from sum of detectors
            self.address_stack_tiff2 = fname2 # Stack from single detector
            self.address_single_tiff3 = fname3 # mux_roi tiff

            # Print to console for debugging
            print(f"Addresses loaded:\n 1. {self.address_stack_tiff1}\n 2. {self.address_stack_tiff2}\n 3. {self.address_single_tiff3}\n")

        except Exception as e:
            print(f"Error during file selection: {e}")
            QMessageBox.warning(self.centralwidget, "Error", f"An error occurred: {e}")

        sample_name = self.lineEdit_4.text()

        address_plot_of_stacks, address_final_plots = start_project(self.project_path,sample_name) # creates folders for data analysis

        self.address_plot_of_stacks = address_plot_of_stacks
        self.address_final_plots = address_final_plots

        """
        Clears all QGraphicsView instances in the given parent objects.
        :param parents: The parent objects containing the QGraphicsView instances.
        """
        for attr_name in dir(self):
            attr = getattr(self, attr_name)
            if isinstance(attr, QtWidgets.QGraphicsView):
                attr.setScene(None)
        print(f"Cleaning graphics")  # Optional debugging line

        ## Disable part of the system so analysis can run again
        # List of widget names to exclude from disabling
        excluded_widgets = [
            'qt_spinbox_lineedit',
            'lineEdit_4',
            'lineEdit_5',
            'lineEdit_70',
            'pushButton',
            'pushButton_12',
            'pushButton_13',
            'pushButton_start_project'
        ]

        # Iterate through all child widgets of the central widget
        for widget in self.centralwidget.findChildren(QtWidgets.QWidget):
            # Check the widget type and name to determine if it should be disabled
            if isinstance(widget, (
                    QtWidgets.QPushButton, QtWidgets.QComboBox, QtWidgets.QLineEdit, QtWidgets.QCheckBox)):
                # Disable widget if it is not in the excluded list
                if widget.objectName() not in excluded_widgets:
                    widget.setEnabled(False)

        #Enable next step
        self.pushButton.setEnabled(True)

    def handle_import_stack_names(self):
        # Get the stack.tiff filenames
        address_stack_tiff1 = self.address_stack_tiff1

        image_filenames = import_stack_names(address_stack_tiff1)
        # Modify filenames: Remove only underscores and the last character from each filename except the last one
        modified_filenames = [
            re.sub(r'_', '', filename)[:-1] for filename in image_filenames[:-1]
        ]

        self.listed_elements = modified_filenames

        # Set comboBox to work and remove the first item before adding new ones
        self.comboBox_element.setEnabled(True)
        # Remove all elements
        self.comboBox_element.clear()
        # Add items one by one to the combo box (excluding the last filename)
        for filename in modified_filenames:
            self.comboBox_element.addItem(filename)  # Add each modified filename to the combo box

        # Optionally, enable the comboBox if needed
        self.comboBox_element.setEnabled(True)
        self.pushButton_2.setEnabled(True)
        self.lineEdit_1.setEnabled(True)
        self.lineEdit_2.setEnabled(True)
        self.lineEdit_3.setEnabled(True)

    def handle_import_file(self):
        # Fetch user input
        address_stack_tiff1 = self.address_stack_tiff1
        address_final_plots =  self.address_final_plots
        detector = self.lineEdit_5.text()
        element = self.comboBox_element.currentText()
        figure_parameters_1 = float(self.lineEdit_1.text()) # x dimension
        figure_parameters_2 = float(self.lineEdit_2.text())  # y dimension
        figure_parameters_3 = float(self.lineEdit_3.text()) # shrink color bar
        figure_parameters = np.array([figure_parameters_1,figure_parameters_2,figure_parameters_3])

        # Generate the plot and get the image path
        img_path, gray_values_pt, gray_values_ir = import_file(address_stack_tiff1,address_final_plots, element, detector,figure_parameters)

        # Store values for later use
        self.gray_values_pt = gray_values_pt
        self.gray_values_ir = gray_values_ir

        scene = QtWidgets.QGraphicsScene()
        renderer = QtSvg.QSvgRenderer(img_path)
        pixmap = QtGui.QPixmap(renderer.defaultSize())
        pixmap.fill(QtCore.Qt.GlobalColor.transparent)  # Clear background if necessary
        painter = QtGui.QPainter(pixmap)
        renderer.render(painter)
        painter.end()
        pixmap_item = QtWidgets.QGraphicsPixmapItem(pixmap)
        scene.addItem(pixmap_item)
        self.graphicsView.setScene(scene)

        self.comboBox_methods_thresh.setEnabled(True)
        self.pushButton_3.setEnabled(True)
        self.lineEdit_6.setEnabled(True)
        self.lineEdit_7.setEnabled(True)
        self.pushButton_clean.setEnabled(True)

    def handle_thresholding_blur(self):
        # Fetch user input
        address_plot_of_stacks = self.address_plot_of_stacks
        element = self.comboBox_element.currentText()
        thresh_method = self.comboBox_methods_thresh.currentText()
        val_pt_manual = float(self.lineEdit_6.text())
        val_ir_manual = float(self.lineEdit_7.text())

        # Generate the plot and get the image path
        img_path_2, img_path_3, img_path_4, val_pt, val_ir, blurred_gray_pt, blurred_gray_ir = thresholding_blur(address_plot_of_stacks,
                                                                                                                 element,
                                                                                                                 self.gray_values_pt,
                                                                                                                 self.gray_values_ir,
                                                                                                                 thresh_method,
                                                                                                                 val_pt_manual,
                                                                                                                 val_ir_manual)

        scene_2 = QtWidgets.QGraphicsScene()
        renderer_2 = QtSvg.QSvgRenderer(img_path_2)
        pixmap_2 = QtGui.QPixmap(renderer_2.defaultSize())
        pixmap_2.fill(QtCore.Qt.GlobalColor.transparent)  # Clear background if necessary
        painter_2 = QtGui.QPainter(pixmap_2)
        renderer_2.render(painter_2)
        painter_2.end()
        pixmap_item_2 = QtWidgets.QGraphicsPixmapItem(pixmap_2)
        scene_2.addItem(pixmap_item_2)
        self.graphicsView_2.setScene(scene_2)

        print("threshold for Pt =", val_pt)
        print("threshold for Ir =", val_ir)

        # Store values for later use
        self.val_pt = val_pt
        self.val_ir = val_ir
        self.blurred_gray_pt = blurred_gray_pt
        self.blurred_gray_ir = blurred_gray_ir

        # Create a scene for GraphicsView_3
        scene_3 = QtWidgets.QGraphicsScene()
        # Combine val_pt and val_ir into a single line to fit the narrow height
        combined_text = f"Pt Threshold: {self.val_pt} | Ir Threshold: {self.val_ir}"
        # Add the text item to the scene
        text_item = QtWidgets.QGraphicsTextItem(combined_text)
        # Adjust font size to fit within the limited vertical space
        font = QtGui.QFont("Arial", 10)  # Small font size for limited height
        text_item.setFont(font)
        # Center the text within the GraphicsView
        text_width = text_item.boundingRect().width()
        text_height = text_item.boundingRect().height()
        text_item.setPos((541 - text_width) / 2, (31 - text_height) / 2)  # Center horizontally and vertically
        # Add the text item to the scene
        scene_3.addItem(text_item)
        # Set the scene for GraphicsView_3
        self.graphicsView_3.setScene(scene_3)

        scene_4 = QtWidgets.QGraphicsScene()
        renderer_3 = QtSvg.QSvgRenderer(img_path_3)
        pixmap_3 = QtGui.QPixmap(renderer_3.defaultSize())
        pixmap_3.fill(QtCore.Qt.GlobalColor.transparent)  # Clear background if necessary
        painter_3 = QtGui.QPainter(pixmap_3)
        renderer_3.render(painter_3)
        painter_3.end()
        pixmap_item_3 = QtWidgets.QGraphicsPixmapItem(pixmap_3)
        scene_4.addItem(pixmap_item_3)
        self.graphicsView_4.setScene(scene_4)

        scene_5 = QtWidgets.QGraphicsScene()
        renderer_4 = QtSvg.QSvgRenderer(img_path_4)
        pixmap_4 = QtGui.QPixmap(renderer_4.defaultSize())
        pixmap_4.fill(QtCore.Qt.GlobalColor.transparent)  # Clear background if necessary
        painter_4 = QtGui.QPainter(pixmap_4)
        renderer_4.render(painter_4)
        painter_4.end()
        pixmap_item_4 = QtWidgets.QGraphicsPixmapItem(pixmap_4)
        scene_5.addItem(pixmap_item_4)
        self.graphicsView_5.setScene(scene_5)

        # enable elements from the next page
        self.lineEdit_8.setEnabled(True)
        self.lineEdit_9.setEnabled(True)
        self.pushButton_4.setEnabled(True)

    def handle_binary_plots(self):

        # Fetch user input
        address_final_plots = self.address_final_plots
        element = self.comboBox_element.currentText()
        detector = self.lineEdit_5.text()
        val_pt = self.val_pt
        val_ir = self.val_ir
        blurred_gray_pt = self.blurred_gray_pt
        blurred_gray_ir = self.blurred_gray_ir
        figure_parameters_blur_1 = float(self.lineEdit_8.text()) # x axis
        figure_parameters_blur_2 = float(self.lineEdit_9.text()) # y axis
        figure_parameters_blur = np.array([figure_parameters_blur_1,figure_parameters_blur_2])

        img_path_5, binary_mask_nafion_DF, binary_mask_cat_DF, binary_mask_nafion, binary_mask_cat = binary_plots(address_final_plots,
                                                                                                                  element,
                                                                                                                  detector,
                                                                                                                  val_pt,
                                                                                                                  val_ir,
                                                                                                                  blurred_gray_pt,
                                                                                                                  blurred_gray_ir,
                                                                                                                  figure_parameters_blur)

        # Store values for later use
        self.binary_mask_nafion_DF = binary_mask_nafion_DF
        self.binary_mask_cat_DF = binary_mask_cat_DF
        self.binary_mask_nafion = binary_mask_nafion
        self.binary_mask_cat = binary_mask_cat

        scene_6 = QtWidgets.QGraphicsScene()
        renderer_5 = QtSvg.QSvgRenderer(img_path_5)
        pixmap_5 = QtGui.QPixmap(renderer_5.defaultSize())
        pixmap_5.fill(QtCore.Qt.GlobalColor.transparent)  # Clear background if necessary
        painter_5 = QtGui.QPainter(pixmap_5)
        renderer_5.render(painter_5)
        painter_5.end()
        pixmap_item_5 = QtWidgets.QGraphicsPixmapItem(pixmap_5)
        scene_6.addItem(pixmap_item_5)
        self.graphicsView_6.setScene(scene_6)

        self.pushButton_5.setEnabled(True)  # enable next functions
        self.lineEdit_11.setEnabled(True)  # Initially disabled
        self.lineEdit_12.setEnabled(True)  # Initially disabled
        self.lineEdit_13.setEnabled(True)  # Initially disabled
        self.lineEdit_14.setEnabled(True)  # Initially disabled
        self.lineEdit_15.setEnabled(True)  # Initially disabled
        self.lineEdit_16.setEnabled(True)  # Initially disabled

    def handle_segment(self):

        # Fetch user input
        address_stack_tiff2 = self.address_stack_tiff2
        address_final_plots = self.address_final_plots
        address_plot_of_stacks = self.address_plot_of_stacks
        element = self.comboBox_element.currentText()
        detector = self.lineEdit_5.text()
        binary_mask_nafion = self.binary_mask_nafion
        binary_mask_cat = self.binary_mask_cat
        listed_elements = self.listed_elements
        color_bar_shrink_plot1 = float(self.lineEdit_11.text())
        color_bar_shrink_plot2 = float(self.lineEdit_12.text())
        color_bar_shrink_plot3 = float(self.lineEdit_13.text())
        color_bar_shrink_plot4 = float(self.lineEdit_14.text())
        figure_parameters_segment_1 = float(self.lineEdit_15.text())
        figure_parameters_segment_2= float(self.lineEdit_16.text())
        figure_parameters_segment = np.array([color_bar_shrink_plot1,color_bar_shrink_plot2,color_bar_shrink_plot3,color_bar_shrink_plot4,figure_parameters_segment_1,figure_parameters_segment_2])


        img_path_6, selection1_DF, selectionPt_DF, selectionIr_DF, gray_values_real_DF, selectionPt, selectionIr, real_im = segment(
            address_stack_tiff2, address_final_plots, address_plot_of_stacks, element, listed_elements, detector, binary_mask_nafion, binary_mask_cat,figure_parameters_segment)

        self.figure_parameters_segment = figure_parameters_segment
        self.selection1_DF = selection1_DF
        self.selectionPt_DF = selectionPt_DF
        self.selectionIr_DF = selectionIr_DF
        self.gray_values_real_DF = gray_values_real_DF
        self.selectionPt = selectionPt
        self.selectionIr = selectionIr
        self.real_im = real_im

        scene_7 = QtWidgets.QGraphicsScene()
        renderer_6 = QtSvg.QSvgRenderer(img_path_6)
        pixmap_6 = QtGui.QPixmap(renderer_6.defaultSize())
        pixmap_6.fill(QtCore.Qt.GlobalColor.transparent)  # Clear background if necessary
        painter_6 = QtGui.QPainter(pixmap_6)
        renderer_6.render(painter_6)
        painter_6.end()
        pixmap_item_6 = QtWidgets.QGraphicsPixmapItem(pixmap_6)
        scene_7.addItem(pixmap_item_6)
        self.graphicsView_7.setScene(scene_7)

        self.pushButton_7.setEnabled(True)  # Initially disabled
        self.checkBox.setEnabled(True)

    def handle_contour(self):
        # Contour before the fine tune ****
        # Fetch user input
        address_plot_of_stacks = self.address_plot_of_stacks
        element = self.comboBox_element.currentText()
        selection1_DF = self.selection1_DF
        selectionPt_DF = self.selectionPt_DF
        selectionIr_DF = self.selectionIr_DF
        selectionPt = self.selectionPt

        biglist_Pt = Pt_catalyst_contour(address_plot_of_stacks, element, selectionPt_DF, selection1_DF)
        biglist_Ir, contour_second_Ir = Ir_catalyst_contour(address_plot_of_stacks, element, selectionIr_DF, selectionPt)
        biglist_nafion = Nafion_membrane_contour(address_plot_of_stacks, element, selection1_DF, contour_second_Ir)

        print("Contour before the fine tune:")
        print(biglist_Pt)
        print(biglist_Ir)
        print(biglist_nafion)

    def toggle_widgets(self):
        # Enable/disable widgets based on checkbox state
        enabled = self.checkBox.isChecked()
        self.pushButton_6.setEnabled(enabled)
        self.lineEdit_10.setEnabled(enabled)
        # Keep Segment and Export buttons always enabled
        self.pushButton_5.setEnabled(True)
        self.pushButton_7.setEnabled(True)
        self.label_12.setEnabled(True)
        self.graphicsView_7.setEnabled(True)
        self.graphicsView_8.setEnabled(True)

    def handle_fine_tune(self):
        # Fetch user input
        address_final_plots = self.address_final_plots
        address_plot_of_stacks = self.address_plot_of_stacks
        element = self.comboBox_element.currentText()
        detector = self.lineEdit_5.text()
        selection1_DF = self.selection1_DF
        selectionPt_DF = self.selectionPt_DF
        selectionIr_DF = self.selectionIr_DF
        selectionPt = self.selectionPt
        selectionIr = self.selectionIr
        fine_tune_var = int(self.lineEdit_10.text())
        real_im = self.real_im
        figure_parameters_fine_tune = self.figure_parameters_segment

        biglist_Pt = Pt_catalyst_contour(address_plot_of_stacks, element, selectionPt_DF, selection1_DF)
        biglist_Ir, contour_second_Ir = Ir_catalyst_contour(address_plot_of_stacks, element, selectionIr_DF, selectionPt)
        biglist_nafion, nafion_membrane_DF = Nafion_membrane_contour(address_plot_of_stacks, element, selection1_DF,
                                                                     contour_second_Ir)
        self.biglist_nafion = biglist_nafion
        self.nafion_membrane_DF = nafion_membrane_DF

        img_path_7, nafion_membrane_DF1 = fine_tune(address_final_plots, element, detector, real_im, nafion_membrane_DF, selectionPt,
                                                    selectionIr, fine_tune_var,figure_parameters_fine_tune)
        self.nafion_membrane_DF1 = nafion_membrane_DF1

        scene_8 = QtWidgets.QGraphicsScene()
        renderer_7 = QtSvg.QSvgRenderer(img_path_7)
        pixmap_7 = QtGui.QPixmap(renderer_7.defaultSize())
        pixmap_7.fill(QtCore.Qt.GlobalColor.transparent)  # Clear background if necessary
        painter_7 = QtGui.QPainter(pixmap_7)
        renderer_7.render(painter_7)
        painter_7.end()
        pixmap_item_7 = QtWidgets.QGraphicsPixmapItem(pixmap_7)
        scene_8.addItem(pixmap_item_7)
        self.graphicsView_8.setScene(scene_8)

        # enable next window elements
        self.pushButton_8.setEnabled(True)
        self.comboBox_legend_position.setEnabled(True)
        self.lineEdit_17.setEnabled(True)
        self.lineEdit_18.setEnabled(True)
        self.lineEdit_19.setEnabled(True)
        self.lineEdit_20.setEnabled(True)
        self.lineEdit_21.setEnabled(True)
        self.lineEdit_22.setEnabled(True)
        self.lineEdit_23.setEnabled(True)
        self.lineEdit_24.setEnabled(True)
        self.lineEdit_25.setEnabled(True)
        self.lineEdit_26.setEnabled(True)
        self.lineEdit_27.setEnabled(True)
        self.lineEdit_28.setEnabled(True)
        self.lineEdit_29.setEnabled(True)
        self.lineEdit_30.setEnabled(True)
        self.lineEdit_31.setEnabled(True)

    def handle_filter_dust(self):
        #Filter
        filter_dust_value = int(self.lineEdit_17.text())
        nafion_membrane_DF1 = self.nafion_membrane_DF1

        #3D PLOT
        set_tick_position_start=int(self.lineEdit_18.text())
        set_tick_position_end=int(self.lineEdit_19.text())
        elevation_view=int(self.lineEdit_20.text())
        azimutal_view=int(self.lineEdit_21.text())

        plot_3d_var = np.array([set_tick_position_start, set_tick_position_end, elevation_view, azimutal_view])

        #Otsu plot
        color_bar_max_membrane_1=float(self.lineEdit_22.text())
        color_bar_min_membrane_1=float(self.lineEdit_23.text())
        color_bar_shrink_membrane_1=float(self.lineEdit_26.text())
        figure_size_x_membrane_1=float(self.lineEdit_24.text())
        figure_size_y_membrane_1=float(self.lineEdit_25.text())

        plot_multi_otsu_var_1= np.array([color_bar_max_membrane_1,color_bar_min_membrane_1,color_bar_shrink_membrane_1,figure_size_x_membrane_1,figure_size_y_membrane_1])

        # Histogram plot
        histogram_limit_y_max_1=float(self.lineEdit_27.text())
        histogram_limit_x_max_1=float(self.lineEdit_28.text())
        histogram_limit_x_min_1=float(self.lineEdit_29.text())
        figure_size_x_histogram_1=float(self.lineEdit_30.text())
        figure_size_y_histogram_1=float(self.lineEdit_31.text())
        legend_loc = self.comboBox_legend_position.currentText()


        plot_hist_otsu_var_1 = np.array([histogram_limit_y_max_1,histogram_limit_x_max_1,histogram_limit_x_min_1,figure_size_x_histogram_1,figure_size_y_histogram_1])

        img_path_8,nafion_membrane_DF2,img_path_9,img_path_10, img_path_11 = filter_dust(nafion_membrane_DF1, filter_dust_value,plot_3d_var,plot_multi_otsu_var_1,plot_hist_otsu_var_1,legend_loc)

        self.nafion_membrane_DF2=nafion_membrane_DF2

        scene_9 = QtWidgets.QGraphicsScene()
        renderer_9 = QtSvg.QSvgRenderer(img_path_8)
        pixmap_9 = QtGui.QPixmap(renderer_9.defaultSize())
        pixmap_9.fill(QtCore.Qt.GlobalColor.transparent)  # Clear background if necessary
        painter_9 = QtGui.QPainter(pixmap_9)
        renderer_9.render(painter_9)
        painter_9.end()
        pixmap_item_9 = QtWidgets.QGraphicsPixmapItem(pixmap_9)
        scene_9.addItem(pixmap_item_9)
        self.graphicsView_13.setScene(scene_9)

        scene_10 = QtWidgets.QGraphicsScene()
        renderer_10 = QtSvg.QSvgRenderer(img_path_9)
        pixmap_10 = QtGui.QPixmap(renderer_10.defaultSize())
        pixmap_10.fill(QtCore.Qt.GlobalColor.transparent)  # Clear background if necessary
        painter_10 = QtGui.QPainter(pixmap_10)
        renderer_10.render(painter_10)
        painter_10.end()
        pixmap_item_10 = QtWidgets.QGraphicsPixmapItem(pixmap_10)
        scene_10.addItem(pixmap_item_10)
        self.graphicsView_9.setScene(scene_10)

        scene_11 = QtWidgets.QGraphicsScene()
        renderer_11 = QtSvg.QSvgRenderer(img_path_10)
        pixmap_11 = QtGui.QPixmap(renderer_11.defaultSize())
        pixmap_11.fill(QtCore.Qt.GlobalColor.transparent)  # Clear background if necessary
        painter_11 = QtGui.QPainter(pixmap_11)
        renderer_11.render(painter_11)
        painter_11.end()
        pixmap_item_11 = QtWidgets.QGraphicsPixmapItem(pixmap_11)
        scene_11.addItem(pixmap_item_11)
        self.graphicsView_11.setScene(scene_11)

        scene_12 = QtWidgets.QGraphicsScene()
        renderer_12 = QtSvg.QSvgRenderer(img_path_11)
        pixmap_12 = QtGui.QPixmap(renderer_12.defaultSize())
        pixmap_12.fill(QtCore.Qt.GlobalColor.transparent)  # Clear background if necessary
        painter_12 = QtGui.QPainter(pixmap_12)
        renderer_12.render(painter_12)
        painter_12.end()
        pixmap_item_12 = QtWidgets.QGraphicsPixmapItem(pixmap_12)
        scene_12.addItem(pixmap_item_12)
        self.graphicsView_10.setScene(scene_12)

        # enable next parameters and plot
        self.pushButton_load_pristine.setEnabled(True)

    def load_pristine(self):
        try:
            # Prompt for the fourth CSV file
            QMessageBox.information(
                self.centralwidget,
                "Select File",
                "Please select the first CSV file - bins for pristine histogram."
            )
            fname4, _ = QFileDialog.getOpenFileName(
                self.centralwidget, 'Open CSV File 1 - normalized bins for pristine',
                'D:/x',
                'CSV Files (*.csv)'
            )

            if not fname4:
                raise Exception("CSV file 1 not selected.")

            file_name4 = os.path.basename(fname4)
            print(f"Selected CSV file 1: {file_name4}")

            # Prompt for the fifth CSV file
            QMessageBox.information(
                self.centralwidget,
                "Select File",
                "Please select the second CSV file - counts for pristine histogram."
            )
            fname5, _ = QFileDialog.getOpenFileName(
                self.centralwidget, 'Open CSV File 2 - normalized counts for pristine',
                'D:/x',
                'CSV Files (*.csv)'
            )

            if not fname5:
                raise Exception("CSV file 2 not selected.")

            file_name5 = os.path.basename(fname5)
            print(f"Selected CSV file 2: {file_name5}")

            # Prompt for the fourth CSV file
            QMessageBox.information(
                self.centralwidget,
                "Select File",
                "Please select the first CSV file - bins for pristine histogram."
            )
            fname6, _ = QFileDialog.getOpenFileName(
                self.centralwidget, 'Open CSV File 3 - normalized bins for quantified pristine',
                'D:/x',
                'CSV Files (*.csv)'
            )

            if not fname6:
                raise Exception("CSV file 3 not selected.")

            file_name6 = os.path.basename(fname6)
            print(f"Selected CSV file 3: {file_name6}")

            # Prompt for the fifth CSV file
            QMessageBox.information(
                self.centralwidget,
                "Select File",
                "Please select the second CSV file - counts for quantified pristine histogram."
            )
            fname7, _ = QFileDialog.getOpenFileName(
                self.centralwidget, 'Open CSV File 4 - counts for quantified pristine histogram',
                'D:/x',
                'CSV Files (*.csv)'
            )

            if not fname7:
                raise Exception("CSV file 4 not selected.")

            file_name7 = os.path.basename(fname7)
            print(f"Selected CSV file 4: {file_name7}")

            # Store addresses into variables
            self.address_csv1 = fname4 # bins for pristine - gray values
            self.address_csv2 = fname5 # counts for pristine - gray values
            self.address_csv3 = fname6  # bins for pristine - quantified
            self.address_csv4 = fname7  # counts for pristine - quantified

            # Print to console for debugging
            print(
                f"Addresses loaded:\n 1. {self.address_csv1}\n 2. {self.address_csv2} 3. {self.address_csv3}\n 4. {self.address_csv4}")

        except Exception as e:
            print(f"Error during file selection: {e}")
            QMessageBox.warning(self.centralwidget, "Error", f"An error occurred: {e}")

        # enable next parameters and plot
        self.pushButton_9.setEnabled(True)
        self.comboBox_histogram_legend.setEnabled(True)
        self.comboBox_membrane_legend.setEnabled(True)
        self.lineEdit_32.setEnabled(True)
        self.lineEdit_33.setEnabled(True)
        self.lineEdit_34.setEnabled(True)
        self.lineEdit_35.setEnabled(True)
        self.lineEdit_36.setEnabled(True)
        self.lineEdit_37.setEnabled(True)
        self.lineEdit_38.setEnabled(True)
        self.lineEdit_39.setEnabled(True)

    def handle_segment_nafion(self):
        # Get main info for directory
        address_final_plots = self.address_final_plots
        address_csv1 = self.address_csv1 # bins for pristine
        address_csv2 = self.address_csv2 # counts for pristine
        acronym = self.lineEdit_70.text()
        detector = self.lineEdit_5.text()
        element = self.comboBox_element.currentText()

        # Figure configuration-------------------------------
        figure_size_x_main_plot_1 = int(self.lineEdit_33.text())
        figure_size_y_main_plot_1 = int(self.lineEdit_32.text())

        plot_main_var_1 = np.array(
            [figure_size_x_main_plot_1,  figure_size_y_main_plot_1])

        # Otsu in the main plot-------------------------------------
        color_bar_min_main_plot_1 = float(self.lineEdit_34.text())
        color_bar_max_main_plot_1 = float(self.lineEdit_36.text())
        color_bar_shrink_main_plot_1 = float(self.lineEdit_35.text())

        plot_main_multi_otsu_var_1 = np.array(
            [ color_bar_min_main_plot_1, color_bar_max_main_plot_1, color_bar_shrink_main_plot_1])

        #3D PLOT-----------------------------------
        set_tick_position_start=int(self.lineEdit_18.text())
        set_tick_position_end=int(self.lineEdit_19.text())
        elevation_view=int(self.lineEdit_20.text())
        azimutal_view=int(self.lineEdit_21.text())

        plot_3d_var_2 = np.array([set_tick_position_start, set_tick_position_end, elevation_view, azimutal_view])

        # Histogram in the main plot------------------------
        main_plot_histogram_limit_x_min_1 = float(self.lineEdit_37.text())
        main_plot_histogram_limit_x_max_1 = float(self.lineEdit_38.text())
        main_plot_histogram_limit_y_max_1 = float(self.lineEdit_39.text())
        legend_loc_1=self.comboBox_histogram_legend.currentText()

        plot_main_histogram_var_1 = np.array(
            [main_plot_histogram_limit_x_min_1,main_plot_histogram_limit_x_max_1,main_plot_histogram_limit_y_max_1])

        # Membrane in the main plot------------------------
        legend_loc_2=self.comboBox_membrane_legend.currentText()

        # global var-----------------------
        nafion_membrane_DF2=self.nafion_membrane_DF2


        #Core function-------------------------
        img_path_12, images = segment_nafion(address_final_plots,address_csv1,address_csv2, element, acronym, detector, nafion_membrane_DF2, plot_main_var_1,plot_3d_var_2, plot_main_multi_otsu_var_1, plot_main_histogram_var_1,legend_loc_1, legend_loc_2)

        self.images= images

        # Display the SVG in QGraphicsView---------------------------
        scene_13 = QtWidgets.QGraphicsScene()
        renderer_13 = QtSvg.QSvgRenderer(img_path_12)
        pixmap_13 = QtGui.QPixmap(renderer_13.defaultSize())
        pixmap_13.fill(QtCore.Qt.GlobalColor.transparent)  # Clear background if necessary
        painter_13 = QtGui.QPainter(pixmap_13)
        renderer_13.render(painter_13)
        painter_13.end()
        pixmap_item_13 = QtWidgets.QGraphicsPixmapItem(pixmap_13)
        scene_13.addItem(pixmap_item_13)
        self.graphicsView_12.setScene(scene_13)

        # Enable all components at the next page
        for line_edit in self.line_edits:
            line_edit.setEnabled(True)

        self.pushButton_10.setEnabled(True)
        self.comboBox_12.setEnabled(True)
        self.comboBox_13.setEnabled(True)

    def handle_thickness(self):
        # Get main info for directory
        address_final_plots = self.address_final_plots
        address_single_tiff3 = self.address_single_tiff3
        acronym = self.lineEdit_70.text()
        element = self.comboBox_element.currentText()
        detector = self.lineEdit_5.text()

        color_bar_min_gray = float(self.lineEdit_40.text())
        color_bar_shrink_gray = float(self.lineEdit_41.text())
        color_bar_max_gray = float(self.lineEdit_42.text())
        gray_scale_normal_plot = np.array([color_bar_min_gray,color_bar_shrink_gray,color_bar_max_gray])

        color_bar_min_thickness = float(self.lineEdit_43.text())
        color_bar_max_thickness = float(self.lineEdit_44.text())
        color_bar_shrink_thickness = float(self.lineEdit_45.text())
        thickness_normal_plot = np.array([color_bar_min_thickness,color_bar_max_thickness,color_bar_shrink_thickness])

        histogram_y_max_gray = float(self.lineEdit_46.text())
        histogram_x_max_gray = float(self.lineEdit_47.text())
        histogram_x_min_gray = float(self.lineEdit_48.text())
        gray_scale_histogram_plot = np.array([histogram_y_max_gray, histogram_x_min_gray,histogram_x_max_gray])


        histogram_x_max_thickness = float(self.lineEdit_49.text())
        histogram_y_max_thickness = float(self.lineEdit_50.text())
        histogram_x_min_thickness = float(self.lineEdit_51.text())
        thickness_histogram_plot = np.array([histogram_y_max_thickness,histogram_x_max_thickness,histogram_x_min_thickness])

        air_absorption_mean = float(self.lineEdit_52.text())
        attenuation_length = float(self.lineEdit_53.text())
        figure_size_x = float(self.lineEdit_54.text())
        figure_size_y = float(self.lineEdit_55.text())
        main_parameters = np.array([air_absorption_mean,attenuation_length,figure_size_x,figure_size_y])

        #nafion_membrane_DF2= self.nafion_membrane_DF2
        nafion_membrane_DF1 = self.nafion_membrane_DF2

        legend_loc_1 = self.comboBox_12.currentText() # gray scale values plot
        legend_loc_2 = self.comboBox_13.currentText() # thickness plot
        legend_loc_var = np.array([legend_loc_1,legend_loc_2])

        img_path_13,gray_values_thickness_DF = thickness(address_final_plots,address_single_tiff3,element,acronym, detector, nafion_membrane_DF1,gray_scale_normal_plot,thickness_normal_plot,gray_scale_histogram_plot,thickness_histogram_plot,legend_loc_var,main_parameters)

        self.gray_values_thickness_DF = gray_values_thickness_DF

        # Display the SVG in QGraphicsView---------------------------
        scene_14 = QtWidgets.QGraphicsScene()
        renderer_14 = QtSvg.QSvgRenderer(img_path_13)
        pixmap_14 = QtGui.QPixmap(renderer_14.defaultSize())
        pixmap_14.fill(QtCore.Qt.GlobalColor.transparent)  # Clear background if necessary
        painter_14 = QtGui.QPainter(pixmap_14)
        renderer_14.render(painter_14)
        painter_14.end()
        pixmap_item_14 = QtWidgets.QGraphicsPixmapItem(pixmap_14)
        scene_14.addItem(pixmap_item_14)
        self.graphicsView_14.setScene(scene_14)

        for line_edit in self.line_edits_tab7:
            line_edit.setEnabled(True)

        self.pushButton_11.setEnabled(True)
        self.comboBox_14.setEnabled(True)

    def handle_quantify(self):
        # Get main info for directory
        address_final_plots = self.address_final_plots
        address_csv3 = self.address_csv3 # bins for pristine - quantified
        address_csv4 = self.address_csv4 # counts for pristine - quantifid
        detector = self.lineEdit_5.text()
        acronym = self.lineEdit_70.text()
        element = self.comboBox_element.currentText()
        images = self.images
        gray_values_thickness_DF = self.gray_values_thickness_DF

        # figure_parameters_quantify
        y_value_quantify = float(self.lineEdit_56.text())  # y
        x_value_quantify = float(self.lineEdit_57.text())  # x
        conversion_factor_quantify = float(self.lineEdit_66.text())  # conversion factor
        figure_parameters_quantify = np.array([y_value_quantify,x_value_quantify,conversion_factor_quantify])

        #figure_3dplot_quantify
        start_value_quantify = int(self.lineEdit_58.text())  # start
        tick_position_end_quantify = int(self.lineEdit_59.text())  # Set tick position: end
        elevation_view_quantify = int(self.lineEdit_60.text())  # elevation view
        azimutal_view_quantify = int(self.lineEdit_61.text())  # azimutal view
        shift_quantify = int(self.lineEdit_62.text()) # intrinsic axis shift for adjust
        figure_3dplot_quantify = np.array([start_value_quantify,tick_position_end_quantify,elevation_view_quantify,azimutal_view_quantify,shift_quantify])

        # figure_histogram_quantify
        histogram_limit_y_quantify = float(self.lineEdit_63.text())  # Histogram limit (y)
        histogram_limit_max_x_quantify = float(self.lineEdit_64.text())  # Histogram limit max (x)
        histogram_limit_min_x_quantify = float(self.lineEdit_65.text())  # Histogram limit min (x)
        figure_histogram_quantify = np.array([histogram_limit_y_quantify,histogram_limit_max_x_quantify,histogram_limit_min_x_quantify])

        # figure_membrane_quantify
        color_bar_min_quantify = float(self.lineEdit_67.text())  # Color bar min (a.u.)
        color_bar_shrink_quantify = float(self.lineEdit_68.text())  # Color bar shrink (0-1)
        color_bar_max_quantify = float(self.lineEdit_69.text())  # Color bar max (a.u.)
        figure_membrane_quantify = np.array([color_bar_min_quantify,color_bar_shrink_quantify,color_bar_max_quantify])

        # Comboxbox legend
        legend_loc_quantify = self.comboBox_14.currentText()  # thickness plot

        img_path_14 = quantify(address_final_plots, address_csv3, address_csv4,detector,element,acronym,images,gray_values_thickness_DF,figure_parameters_quantify,figure_3dplot_quantify,figure_histogram_quantify,figure_membrane_quantify,legend_loc_quantify)

        scene_15 = QtWidgets.QGraphicsScene()
        renderer_15 = QtSvg.QSvgRenderer(img_path_14)
        pixmap_15 = QtGui.QPixmap(renderer_15.defaultSize())
        pixmap_15.fill(QtCore.Qt.GlobalColor.transparent)  # Clear background if necessary
        painter_15 = QtGui.QPainter(pixmap_15)
        renderer_15.render(painter_15)
        painter_15.end()
        pixmap_item_15 = QtWidgets.QGraphicsPixmapItem(pixmap_15)
        scene_15.addItem(pixmap_item_15)
        self.graphicsView_15.setScene(scene_15)


        # Create a scene for GraphicsView_16
        scene_16 = QtWidgets.QGraphicsScene()
        # Text
        combined_text = f"Element: {element} | Detector: {detector} | Conversion factor used: {conversion_factor_quantify:.2e}"
        # Add the text item to the scene
        text_item = QtWidgets.QGraphicsTextItem(combined_text)
        # Adjust font size to fit within the limited vertical space
        font = QtGui.QFont("Arial", 10)  # Small font size for limited height
        text_item.setFont(font)
        # Center the text within the GraphicsView
        text_width = text_item.boundingRect().width()
        text_height = text_item.boundingRect().height()
        text_item.setPos((541 - text_width) / 2, (31 - text_height) / 2)  # Center horizontally and vertically
        # Add the text item to the scene
        scene_16.addItem(text_item)
        # Set the scene for GraphicsView_3
        self.graphicsView_16.setScene(scene_16)

    def handle_save_varibles(self):

        # Get the sample name from the QLineEdit
        sample_name = self.lineEdit_4.text()

        # Default file path with the sample name and "_parameters.csv"
        file_path_var = f"D:/x/variables/{sample_name}_parameters.csv"

        # Open save file dialog with the initial file path
        fname, _ = QFileDialog.getSaveFileName(self.centralwidget, 'Save File', file_path_var, 'CSV Files (*.csv)')

        if fname:  # If the user selected a file path (not canceled)
            # Ensure the file has the .csv extension
            if not fname.endswith('.csv'):
                fname += '.csv'

            # Assuming `names_after_equal` contains the variable names
            names_after_equal = [
                "figure_parameters_1", "figure_parameters_2", "figure_parameters_3",
                "sample_name", "detector", "val_pt_manual", "val_ir_manual",
                "figure_parameters_blur_1", "figure_parameters_blur_2", "fine_tune_var",
                "color_bar_shrink_plot1", "color_bar_shrink_plot2", "color_bar_shrink_plot3",
                "color_bar_shrink_plot4", "figure_parameters_segment_1", "figure_parameters_segment_2",
                "filter_dust_value", "set_tick_position_start", "set_tick_position_end",
                "elevation_view", "azimutal_view", "color_bar_max_membrane_1", "color_bar_min_membrane_1",
                "figure_size_x_membrane_1", "figure_size_y_membrane_1", "color_bar_shrink_membrane_1",
                "histogram_limit_y_max_1", "histogram_limit_x_max_1", "histogram_limit_x_min_1",
                "figure_size_x_histogram_1", "figure_size_y_histogram_1", "figure_size_y_main_plot_1",
                "figure_size_x_main_plot_1", "color_bar_min_main_plot_1", "color_bar_shrink_main_plot_1",
                "color_bar_max_main_plot_1", "main_plot_histogram_limit_x_min_1",
                "main_plot_histogram_limit_x_max_1", "main_plot_histogram_limit_y_max_1",
                "color_bar_min_gray", "color_bar_shrink_gray", "color_bar_max_gray",
                "color_bar_min_thickness", "color_bar_max_thickness", "color_bar_shrink_thickness",
                "histogram_y_max_gray", "histogram_x_max_gray", "histogram_x_min_gray",
                "histogram_x_max_thickness", "histogram_y_max_thickness", "histogram_x_min_thickness",
                "air_absorption_mean", "attenuation_length", "figure_size_x", "figure_size_y",
                "y_value_quantify", "x_value_quantify", "conversion_factor_quantify",
                "start_value_quantify", "tick_position_end_quantify", "elevation_view_quantify",
                "azimutal_view_quantify", "shift_quantify", "histogram_limit_y_quantify",
                "histogram_limit_max_x_quantify", "histogram_limit_min_x_quantify",
                "color_bar_min_quantify", "color_bar_shrink_quantify", "color_bar_max_quantify","acronym"
            ]

            # Collect all QLineEdit widgets
            all_line_edits = self.centralwidget.findChildren(QtWidgets.QLineEdit)
            all_line_edits = [line_edit for line_edit in all_line_edits if isinstance(line_edit, QtWidgets.QLineEdit)]

            try:
                # Sort line edits based on the numeric index in their objectName (lineEdit_x)
                all_line_edits.sort(key=lambda line_edit: (
                    int(re.search(r'(\d+)', line_edit.objectName()).group(1)) if re.search(r'(\d+)',
                                                                                           line_edit.objectName()) else float(
                        'inf')
                ))

                # Create an OrderedDict to map lineEdit_x values to names_after_equal
                mapped_values = OrderedDict()

                # Iterate through sorted line edits and map their values to corresponding variable names
                for i, line_edit in enumerate(all_line_edits):
                    if i < len(names_after_equal):
                        name = names_after_equal[i]
                        line_edit_name = line_edit.objectName()
                        line_edit_value = line_edit.text()

                        # Map the variable name to its value
                        mapped_values[name] = {
                            "lineEdit": line_edit_name,
                            "value": line_edit_value
                        }

                # Save to the selected CSV file
                with open(fname, mode="w", newline="") as file:
                    writer = csv.writer(file)
                    # Write the header row
                    writer.writerow(["Variable Name", "LineEdit Name", "Value"])
                    # Write the data rows
                    for name, info in mapped_values.items():
                        writer.writerow([name, info["lineEdit"], info["value"]])

                # Inform the user that the file was saved successfully
                QMessageBox.information(self.centralwidget, "Success", f"Parameters saved successfully ")

            except Exception as e:
                print(f"Error during mapping: {e}")
                QMessageBox.warning(self.centralwidget, "Error", "An error occurred while saving the file.")

    def handle_load_variables(self):
        # Open file dialog to choose a file
        fname, _ = QFileDialog.getOpenFileName(self.centralwidget, 'Open File',
                                               'D:/x/variables',
                                               'CSV Files (*.csv)')

        if fname:  # If a file was selected (not canceled)
            # Extract the file name from the full file path
            file_name = os.path.basename(fname)

            # Example: Set the file name into a QLabel or QLineEdit in the UI
            self.filename_loaded = file_name

            # Print the file name to the console
            print(f"Selected file: {self.filename_loaded}")
            print(f"\n")
            # Open and read the CSV file
            try:
                with open(fname, mode='r') as file:
                    reader = csv.reader(file)
                    # Skip the header row
                    next(reader)

                    # Create a dictionary from the CSV rows (Variable Name, LineEdit Name, Value)
                    loaded_data = {row[1]: row[2] for row in reader}


                    # Clear all QLineEdits before setting new values
                    all_line_edits = self.centralwidget.findChildren(QLineEdit)  # Only find QLineEdits
                    for line_edit in all_line_edits:
                        line_edit.clear()  # Clear the content of each QLineEdit

                    # Map the loaded values to the appropriate QLineEdits
                    for name, value in loaded_data.items():
                        for line_edit in all_line_edits:
                            if line_edit.objectName() == name:  # Match the lineEdit name with the variable name
                                line_edit.setText(value)  # Set the value in the corresponding lineEdit

                    QMessageBox.information(self.centralwidget, "Success",
                                            f"Variables loaded successfully from {file_name}")

                    ## Disable part of the system so analysis can run again
                    # List of widget names to exclude from disabling
                    excluded_widgets = [
                        'qt_spinbox_lineedit',
                        'lineEdit_4',
                        'lineEdit_5',
                        'lineEdit_70',
                        'pushButton',
                        'pushButton_12',
                        'pushButton_13',
                        'pushButton_start_project'
                    ]

                    # Iterate through all child widgets of the central widget
                    for widget in self.centralwidget.findChildren(QtWidgets.QWidget):
                        # Check the widget type and name to determine if it should be disabled
                        if isinstance(widget, (
                        QtWidgets.QPushButton, QtWidgets.QComboBox, QtWidgets.QLineEdit, QtWidgets.QCheckBox)):
                            # Disable widget if it is not in the excluded list
                            if widget.objectName() not in excluded_widgets:
                                widget.setEnabled(False)


            except Exception as e:
                print(f"Error during loading: {e} \n")
                QMessageBox.warning(self.centralwidget, "Error", "An error occurred while loading the file.")

        """
        Clears all QGraphicsView instances in the given parent objects.
        :param parents: The parent objects containing the QGraphicsView instances.
        """
        for attr_name in dir(self):
            attr = getattr(self, attr_name)
            if isinstance(attr, QtWidgets.QGraphicsView):
                attr.setScene(None)
        print(f"Cleaning graphics")  # Optional debugging line

    # Wigdets -----------------------------
    def retranslateUi(self, MainWindow):
        app.setStyle("Fusion")
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "PEM 2D XRF"))
        self.label.setText(_translate("MainWindow", "Directories"))
        self.label_1.setText(_translate("MainWindow", "Figure size (x)"))
        self.label_1_1.setText(_translate("MainWindow", "Figure size (y)"))
        self.label_1_2.setText(_translate("MainWindow", "Color bar shrink (0-1)"))
        self.label_1_3.setText(_translate("MainWindow", "Parameters"))
        self.label_1_4.setText(_translate("MainWindow", "Acronym"))
        self.label_2.setText(_translate("MainWindow", "Day"))
        self.label_3.setText(_translate("MainWindow", "Sample name"))
        self.label_4.setText(_translate("MainWindow", "Detector"))
        self.label_5.setText(_translate("MainWindow", "Element"))
        self.pushButton.setText(_translate("MainWindow", "Load sample"))
        self.pushButton_2.setText(_translate("MainWindow", "Import"))
        self.pushButton_12.setText(_translate("MainWindow", "Save"))
        self.pushButton_13.setText(_translate("MainWindow", "Load"))
        self.pushButton_start_project.setText(_translate("MainWindow", "Start project"))
        self.pushButton_clean.setText(_translate("MainWindow", "Clean"))
        self.tabWidget.setTabText(self.tabWidget.indexOf(self.tab), _translate("MainWindow", "Importing file"))

        self.pushButton_3.setText(_translate("MainWindow", "Plot"))
        self.label_6.setText(_translate("MainWindow", "Threshold method"))
        self.label_7.setText(_translate("MainWindow", "Pt threshold"))
        self.label_8.setText(_translate("MainWindow", "Ir threshold"))
        self.label_9.setText(_translate("MainWindow", "Manual Thresholding"))
        self.label_10.setText(_translate("MainWindow", "Pt histogram"))
        self.label_11.setText(_translate("MainWindow", "Ir histogram"))
        self.tabWidget.setTabText(self.tabWidget.indexOf(self.tab_2), _translate("MainWindow", "Thresholding sample"))

        self.label_11_1.setText(_translate("MainWindow", "Figure size (x)"))
        self.label_11_2.setText(_translate("MainWindow", "Figure size (y)"))
        self.pushButton_4.setText(_translate("MainWindow", "Generate binary mask"))
        self.tabWidget.setTabText(self.tabWidget.indexOf(self.tab_3),
                                  _translate("MainWindow", "Binary mask for coordinates"))

        self.pushButton_5.setText(_translate("MainWindow", "Segment"))
        self.pushButton_6.setText(_translate("MainWindow", "Fine tune"))
        self.pushButton_7.setText(_translate("MainWindow", "Export coordinates to ImJ"))
        self.label_12.setText(_translate("MainWindow", "Fine tune (Px)"))
        self.label_12_1.setText(_translate("MainWindow", "Color bar shrink (0-1) plot 1"))
        self.label_12_2.setText(_translate("MainWindow", "Color bar shrink (0-1) plot 2"))
        self.label_12_3.setText(_translate("MainWindow", "Color bar shrink (0-1) plot 3"))
        self.label_12_4.setText(_translate("MainWindow", "Color bar shrink (0-1) plot 4"))
        self.label_12_5.setText(_translate("MainWindow", "Figure size (x)"))
        self.label_12_6.setText(_translate("MainWindow", "Figure size (y)"))
        self.checkBox.setText(_translate("MainWindow", "Proceed with fine tune"))
        self.tabWidget.setTabText(self.tabWidget.indexOf(self.tab_4), _translate("MainWindow", "Layers and fine tune"))

        self.label_13.setText(_translate("MainWindow", "Filter for dust (a.u.)"))
        self.pushButton_8.setText(_translate("MainWindow", "Apply filter"))
        self.pushButton_9.setText(_translate("MainWindow", "Nafion segmentation"))
        self.pushButton_load_pristine.setText(_translate("MainWindow", "Load pristine histogram"))
        self.label_14.setText(_translate("MainWindow", "Set tick position: start"))
        self.label_15.setText(_translate("MainWindow", "Set tick position: end"))
        self.label_16.setText(_translate("MainWindow", "Color bar min. (a.u.)"))
        self.label_17.setText(_translate("MainWindow", "Color bar max. (a.u.)"))
        self.label_18.setText(_translate("MainWindow", "Figure size (x)"))
        self.label_19.setText(_translate("MainWindow", "Figure size (y)"))
        self.label_20.setText(_translate("MainWindow", "Color bar shrink (0-1)"))
        self.label_21.setText(_translate("MainWindow", "Elevation view:"))
        self.label_22.setText(_translate("MainWindow", "Azimutal view:"))
        self.label_23.setText(_translate("MainWindow", "Figure size (y)"))
        self.label_24.setText(_translate("MainWindow", "Figure size (x)"))
        self.label_25.setText(_translate("MainWindow", "Color bar min. (a.u.)"))
        self.label_26.setText(_translate("MainWindow", "Color bar max. (a.u.)"))
        self.label_27.setText(_translate("MainWindow", "Color bar shrink (0-1)"))
        self.label_28.setText(_translate("MainWindow", "Histogram limit (y)"))
        self.label_29.setText(_translate("MainWindow", "Histogram limit (x)"))
        self.label_30.setText(_translate("MainWindow", "Histogram"))
        self.label_31.setText(_translate("MainWindow", "Membrane"))
        self.label_32.setText(_translate("MainWindow", "Histogram limit (x)"))
        self.label_33.setText(_translate("MainWindow", "Histogram limit (y)"))
        self.label_34.setText(_translate("MainWindow", "Figure size (y)"))
        self.label_35.setText(_translate("MainWindow", "Figure size (x)"))
        self.label_36.setText(_translate("MainWindow", "Legend position"))
        self.tabWidget.setTabText(self.tabWidget.indexOf(self.tab_5), _translate("MainWindow", "Nafion segmentation"))

        self.label_37.setText(_translate("MainWindow", "Color bar shrink (0-1)"))
        self.label_38.setText(_translate("MainWindow", "Color bar min. (a.u.)"))
        self.label_39.setText(_translate("MainWindow", "Color bar max. (a.u.)"))
        self.label_40.setText(_translate("MainWindow", "Color bar max. (a.u.)"))
        self.label_41.setText(_translate("MainWindow", "Color bar min. (a.u.)"))
        self.label_42.setText(_translate("MainWindow", "Color bar shrink (0-1)"))
        self.label_43.setText(_translate("MainWindow", "Gray scale values"))
        self.label_44.setText(_translate("MainWindow", "Air absorption intensity mean value "))
        self.label_45.setText(_translate("MainWindow", "Thickness values"))
        self.label_46.setText(_translate("MainWindow", "Gray scale values"))
        self.label_47.setText(_translate("MainWindow", "Histogram limit (y)"))
        self.label_48.setText(_translate("MainWindow", "Histogram limit (x)"))
        self.label_49.setText(_translate("MainWindow", "Legend position"))
        self.label_50.setText(_translate("MainWindow", "Thickness values"))
        self.label_51.setText(_translate("MainWindow", "Histogram limit (x)"))
        self.label_52.setText(_translate("MainWindow", "Histogram limit (y)"))
        self.label_53.setText(_translate("MainWindow", "Legend position"))
        self.label_54.setText(_translate("MainWindow", "X-Ray attenuation length for Nafion"))
        self.label_55.setText(_translate("MainWindow", "Figure size (x)"))
        self.label_56.setText(_translate("MainWindow", "Figure size (y)"))
        self.label_57.setText(_translate("MainWindow", "Gray scale values"))
        self.pushButton_10.setText(_translate("MainWindow", "Calculate thickness"))
        self.tabWidget.setTabText(self.tabWidget.indexOf(self.tab_6), _translate("MainWindow", "Thickness"))

        self.label_58.setText(_translate("MainWindow", "Figure size (y)"))
        self.label_59.setText(_translate("MainWindow", "Figure size (x)"))
        self.label_60.setText(_translate("MainWindow", "Set tick position: end"))
        self.label_61.setText(_translate("MainWindow", "Azimutal view:"))
        self.label_62.setText(_translate("MainWindow", "Elevation view:"))
        self.label_63.setText(_translate("MainWindow", "Set tick position: start"))
        self.label_63_1.setText(_translate("MainWindow", "Axis relative correction"))
        self.label_64.setText(_translate("MainWindow", "Histogram limit (y)"))
        self.label_65.setText(_translate("MainWindow", "Histogram limit (x)"))
        self.label_66.setText(_translate("MainWindow", "Legend position"))
        self.label_67.setText(_translate("MainWindow", "Conversion factor "))
        self.label_68.setText(_translate("MainWindow", "Color bar shrink (0-1)"))
        self.label_69.setText(_translate("MainWindow", "Color bar min (a.u.)"))
        self.label_70.setText(_translate("MainWindow", "Color bar max (a.u.)"))
        self.pushButton_11.setText(_translate("MainWindow", "Quantify"))
        self.label_71.setText(_translate("MainWindow", "Membrane plot"))
        self.label_72.setText(_translate("MainWindow", "Histogram"))
        self.label_73.setText(_translate("MainWindow", "3D plot"))
        self.tabWidget.setTabText(self.tabWidget.indexOf(self.tab_7), _translate("MainWindow", "Quantify"))


if __name__ == "__main__":
    import sys

    app = QtWidgets.QApplication(sys.argv)
    MainWindow = QtWidgets.QMainWindow()
    ui = Ui_MainWindow()
    ui.setupUi(MainWindow)
    MainWindow.show()
    sys.exit(app.exec())
