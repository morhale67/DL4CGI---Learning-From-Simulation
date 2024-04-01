# DL4CGI - Learning From Simulation
Implementation of the paper: "Learning from simulation: An end-to-end deep-learning approach for computational ghost imaging"

Link to the paper:
https://opg.optica.org/oe/fulltext.cfm?uri=oe-27-18-25560&id=417106


### Project Overview

This project aims to implement image reconstruction using ghost imaging techniques, with a focus on the MNIST dataset. The network's objective is to reconstruct objects captured in ghost imaging simulations. Evaluation metrics such as Mean Squared Error (MSE), Peak Signal-to-Noise Ratio (PSNR), and Structural Similarity Index (SSIM) are utilized to measure the network's performance.

### Implementation Details

- **Ghost Imaging Simulation:** 
  - A simulation for ghost imaging is implemented, and the simulated measurements serve as input to the neural network.

- **Network Evaluation:**
  - The network's performance is evaluated using MSE, PSNR, and SSIM metrics, comparing the reconstructed images with ground truth.

- **Parameterization:**
  - Parameters for the project, including batch size, image dimensions, and number of samples, can be configured in the `Params.py` file.

### Execution

- **main File:**
  - The project execution is orchestrated through the `main.py` file.
  - Users can choose to execute a single run or perform parameter search, which triggers multiple runs.

- **Results:**
  - Each run generates a dedicated folder within the `results` directory.
    - **Log File:** Contains run parameters and details any errors encountered during execution.
    - **Graph Files:** Graphs displaying MSE, SSIM, and PSNR over epochs are saved.
    - **numerical_outputs.pkl File:** Stores graph values, facilitating restoration during design changes.
    - **Train/Test Folders:** Contains images from each epoch during the run.

### Running the Project

1. Set Parameters:
   - Customize parameters such as batch size, image dimensions, etc., in the `Params.py` file.

2. Execution:
   - Run the project from the `main.py` file.
   - Choose between single runs or parameter search.
