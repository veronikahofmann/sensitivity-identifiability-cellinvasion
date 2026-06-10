# sensitivity-identifiability-cellinvasion

This project was developed in line with my Master's Thesis and Sebastian Doctor's Bachelor's Thesis about parameter sensitivity and identifiability in cell invasion models. 

To use the code in this repository, you need Python 3.9 as well as MATLAB 2025a or newer. It possibly also works with older versions, but we did not test it with them.

The folder "models" contains the implementations of the three models investigated in our work.
Most files in that folder refer to the extension of the model from René F. M. van Oers et al., "Mechanical Cell-Matrix Feedback Explains Pairwise and Collective Endothelial Cell Behavior In Vitro," PLOS Computational Biology vol. 10, no. 8, 2014. https://doi.org/10.1371/journal.pcbi.1003774.
The simulation can be started by executing the file `main.py`. It will use the parameters specified in `parameters.py` and produce some animated mp4-plots as well as a line plot of the cell- and ECM-concentration. The plots will be saved as pngs in a folder named "PLOTS", the results will be saved as .out-files in a folder called "RESULTS".
Additionally to the CPM-FEM model, the modules `colsons.py` and `crossleys.py` provide implementations and plotting functions of Chloé Colson's (from C. Colson et al., "Travelling-wave analysis of a model of tumour invasion with degenerate, cross-dependent diffusion," Proceedings. Mathematical, physical, and engineering sciences, vol. 477, no. 2256, p. 20210593, 2021. https://doi.org/10.1098/rspa.2021.0593) and Rebecca Crossley's parameterized PDE models (from R. M. Crossley et al., "Travelling waves in a coarse-grained model of volume-filling cell invasion: Simulations and comparisons," Studies in Applied Mathematics, vol. 151, pp. 1471-1497, 2023. https://doi.org/10.1111/sapm.12635), respectively.

The directory "Sensitivity Analysis" contains the Python scripts for performing the sensitivity analysis of the given models and plotting the resulting sensitivity indices.

In "Naive Fits" the Python scripts for the naive data fit can be found.

Finally, the files in the "Profile Likelihood" folder contain the MATLAB scripts for plotting the results of the parameter identifiability analysis based on Rebecca Crossley's PDE model cited above as well as all plots considered for the analysis. Running the script `create_plots_template.m` in the "Plots" subfolder creates plots for a single profiled parameter component with various different settings, whereas running the script `create_plots_select.m` creates plots for all parameter components with fixed settings. The save directory of the plots should be manually adjusted when running the scripts locally. The settings considered as well as the parameters to be profiled can also be adjusted manually.

You can find my contact details at https://www.math.cit.tum.de/math/personen/wissenschaftliches-personal/hofmann-veronika/.
