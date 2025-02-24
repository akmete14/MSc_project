# Distribution Generalization in Dynamical Real-World Systems
This git implements some pooling and domain generalization methods, applied to the FLUXNET data. The methods implemented are the following
1. Linear Regression
2. Extreme Gradient Boosting (XGBoost)
3. Long Short-Term Memory (LSTM)
4. Invariant Risk Minimization (IRM)
5. Stabilized Regression (SR)

To reproduce the experiments, ensure you have access to the FLUXNET data. Then, clone this repository into your preffered code editor.
## Loading Data and preprocessing
As a first step, upload the data and store it in the main directory in a folder called "Data". Then, put it into your list in .gitignore so that git won't track the data.
