# Distribution Generalization in Dynamical Real-World Systems
This git implements some pooling and domain generalization methods, applied to the FLUXNET data. The methods implemented are the following
1. Linear Regression
2. Extreme Gradient Boosting (XGBoost)
3. Long Short-Term Memory (LSTM)
4. Invariant Risk Minimization (IRM)
5. Stabilized Regression (SR)

To reproduce the experiments, ensure you have access to the FLUXNET data. Then, clone this repository into your preffered code editor.

## Structure of the git
This git implements the above listed methods. It assumes that data is uploaded into the main directory in a folder called "Data". The folder "preprocessing" applies the necessary preprocessing steps to be able to use the data. If you are interested in predicting other variables than just the GPP, you may change the preprocessing file so that it includes the variable you are interested in.
For every method, there is corresponding folder in the main directoy. It contains the three different Set-Ups **In-Site**, **LOSO** and **LOGO**. Every Set-Up contains the **python** file for the implementation, a **shell** file for scheduling the jobs and the **csv** saving the results.


## Loading Data and preprocessing
As a first step, upload the data and store it in the main directory in a folder called "Data". Then, put it into your list in .gitignore so that git won't track the data.
