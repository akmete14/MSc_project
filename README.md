# Distribution Generalization in Dynamical Real-World Systems
This git implements some pooling and domain generalization methods, applied to the FLUXNET data. The methods implemented are the following
1. Linear Regression,
2. Extreme Gradient Boosting (XGBoost),
3. Long Short-Term Memory (LSTM),
4. Invariant Risk Minimization (IRM),
5. Stabilized Regression (SR).

To reproduce the experiments, ensure you have access to the FLUXNET data. Then, clone this repository into your preffered code editor and, in the main directory, create a folder called "Data". Then, upload the data into this folder. Due to the large amount of data, you need to make sure that the "Data" folder is listed in .gitignore. This ensures that you don't upload large amount of data into the git when pushing it.

## Structure of the git
This git implements the above listed methods. It assumes that data is uploaded into the main directory in a folder called "Data". For every method, there is corresponding folder in the main directoy. It contains the three different Set-Ups **In-Site**, **LOSO** and **LOGO**. Every Set-Up contains the **python** file for the implementation, a **shell** file for scheduling the jobs and a **csv**  file with the results. Observe that mostly, the code was run in parallel so save time. Therefore, there is sometimes a python file **merge** included, merging all CSVs to one total csv.

## Setting up a virtual environment
The cleanest way to be able to run the experiments is to use a virtual environment. In order to run the experiments, you need to consider creating a virtual environment. A list of the dependencies needed for running the experiments can be found in the **requirements.txt** file. Use this file to create the virtual environment.
## Running your own methods

### Loading Data and preprocessing
As a first step, upload the data and store it in the main directory in a folder called "Data". Then, put it into your list in .gitignore so that git won't track the data.
