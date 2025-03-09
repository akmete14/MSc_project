# Distribution Generalization in Dynamical Real-World Systems
This git implements some pooling and domain generalization methods, applied to the FLUXNET dataset. The methods implemented are the following
1. Linear Regression (LR),
2. Extreme Gradient Boosting (XGBoost),
3. Long Short-Term Memory (LSTM),
4. Invariant Risk Minimization (IRM),
5. Stabilized Regression (SR with LR and XGB as underlying regressor).

To reproduce the experiments, ensure that you have access to the FLUXNET data. Next, clone this repository. 
<!--
In the main directory, create a folder called "Data" and copy the data into this folder. Due to the large amount of data, this can take some time. Make sure that the "Data" folder is listed in **.gitignore** to ensure that you don't upload large amounts of data when pushing changes to the repository.
-->
## Structure of the git
This git implements the methods listed above. It assumes that data is copied into the main directory in a folder called "Data". For every method, there is a corresponding folder in the main directory. It contains three different setups **In-Site**, **LOSO** and **LOGO**. Each setup contains a **python** file for the implementation of the method in the corresponding setting, a **shell** file for scheduling the jobs, and a **csv**  file with the results. Shell files are used for scheduling jobs. However, if you are not working with scheduling, I still recommend to look into the shell file to see how many resources are necessary so that the job works. The resources requested in the shell file are not optimal in the sense that the code might also work with fewer resources. However, requesting the same resources will surely execute the job successfully.

## Guide on how to reproduce the experiments
### Setting up a virtual environment
The cleanest way to be able to run the experiments is by creating a virtual environment. A list of the dependencies needed in this environment can be found in the **requirements.txt** file. Use this file to create the virtual environment.
<!--
After creating the virtual environment, we are set to reproduce the experiments. By activating the virtual environment, the scripts can be executed in the terminal. If you want to schedule the jobs, just submit the shell file by typing `$ sbatch script.sh` into the terminal, which will submit the job.

Next, I explain how you can run your own experiments on the FLUXNET data. I will show it in an example of the In-Site experimental setting.
-->
## Run your own methods
### Preprocessing data
After having uploaded the data and created the virtual environment, the first step to run your own method is to consider preprocessing of the data.
The implementation of the preprocessing can be found in `.preprocessing/preprocessing.py`. The file reads in the data and executes the preprocessing steps as discussed in Chapter 2. If you are also interested in predicting other fluxes next to the GPP, you can adjust the Python file accordingly. For example, if you want to consider the NEE as target variable, then include it in the **initialize_dataframe** function. Also don't forget to include the corresponding quality control variable NEE_qc, which will be used later when filtering for good quality data.

Below is an example snippet from `./preprocessing/preprocessing.py`, that includes the NEE:

```python
def initialize_dataframe(file1, file2, file3, path):
    # Open data
    ds = xr.open_dataset(path + file1, engine='netcdf4')
    ds = ds[['GPP','GPP_qc','NEE', 'NEE_qc','longitude','latitude']]
    dr = xr.open_dataset(path + file2, engine='netcdf4')
    dr = dr[['Tair','Tair_qc','vpd','vpd_qc','SWdown','SWdown_qc','LWdown','LWdown_qc','SWdown_clearsky','IGBP_veg_short']]
    dt = xr.open_dataset(path + file3, engine='netcdf4')
    dt = dt[['LST_TERRA_Day','LST_TERRA_Night','EVI','NIRv','NDWI_band7','LAI','fPAR']]
```
After choosing the preprocessing of the data, you can start setting up the In-Site experiment.
### Reading the data and defining features & target variable
Let's look at `./LR/In_Site/insite.py` as an example on how to use a regression method from sklearn for the In-Site extrapolation task. First, read in the data which we preprocessed and define the feature set and target variable:
```python
# Load and preprocess data
df = pd.read_csv('/cluster/project/math/akmete/MSc/preprocessing/df_balanced_groups_onevegindex.csv')
df = df.fillna(0)
# Drop unnecessary columns
df = df.drop(columns=['Unnamed: 0','cluster'])
# Convert numerical features from float64 to float32 to save memory
for col in tqdm(df.select_dtypes(include=['float64']).columns, desc="Casting columns"):
    df[col] = df[col].astype('float32')

# Define features and target
feature_columns = [col for col in df.columns if col not in ['GPP', 'site_id']]
target_column = "GPP"

# Initialize result dict
results = {}

# Define the sites to be processed (for sequential execution)
sites_to_process = df['site_id'].unique()
```
Given all sitenames (from the site_id column of the dataframe from the preprocessing), we can now consider the for loop. That is, for every site we do the following: First we split the site data into a chronological 80/20 train/test split
```python
# Extract data belonging to this site
group = df[df['site_id'] == site]
    
# Perform an 80/20 chronological split
n_train = int(len(group) * 0.8)
train = group.iloc[:n_train]
test  = group.iloc[n_train:]
```
Next, we define train and test of the features and the target variable and scale everything
```python
# Extract features and target variables
X_train = train[feature_columns]
y_train = train[target_column]
X_test  = test[feature_columns]
y_test  = test[target_column]
    
# Scale features using MinMaxScaler
scaler_X = MinMaxScaler()
X_train_scaled = scaler_X.fit_transform(X_train)
X_test_scaled  = scaler_X.transform(X_test)
    
# Scale target variable based on training data values
y_train_min = y_train.min()
y_train_max = y_train.max()
if y_train_max - y_train_min == 0:
    y_train_scaled = y_train.values
    y_test_scaled = y_test.values
else:
    y_train_scaled = (y_train - y_train_min) / (y_train_max - y_train_min)
    y_test_scaled  = (y_test - y_train_min) / (y_train_max - y_train_min)

```
Finally, we define the regressor, fit the model, and evaluate using different metrics
```python
# Train a linear regression model on the scaled training data
model = LinearRegression()
model.fit(X_train_scaled, y_train_scaled)
    
# Evaluate the model on the scaled test data
y_pred_scaled = model.predict(X_test_scaled)
mse = mean_squared_error(y_test_scaled, y_pred_scaled)
rmse = np.sqrt(mse)
r2 = r2_score(y_test_scaled, y_pred_scaled)
relative_error = np.mean(np.abs(y_test_scaled - y_pred_scaled) / np.abs(y_test_scaled))
mae = np.mean(np.abs(y_test_scaled - y_pred_scaled))

# Store the model and performance metric for the site
results[site] = {'model': model, 'mse': mse, 'rmse': rmse, 'r2_score': r2, 'relative_error': relative_error, 'mae': mae}    
print(f"Site {site}: MSE = {mse:.6f}")
```
The LinearRegressor() can in principle be replaced by any regressor from sklearn. If you want to implement a method based on Pytorch, I recommend looking at the pipelines of the IRM implementations. If you are considering to implement a tensorflow-based method, then you can follow the pipeline for the LSTM. Given the Python file, you can run it using a shell file for scheduling.
### Shell file
The shell files are important when you work on clusters which require scheduling jobs. In a shell file, you first specify some parameters, such as time and memory, that you need to run the job. Moreover, you specify the modules you need to load so that the packages you import in the Python scripts are loaded appropriately. In addition to the modules, you also activate the virtual environment that you created. In the end of the shell file, you express the command which you want to be executed, for example `python path/to/your/script.py`. A shell file can look like this
```sh
#!/bin/bash
# Sample shell script for SLURM job submission

#SBATCH --job-name=my_job         # Job name
#SBATCH --output=output.log       # Standard output log
#SBATCH --error=error.log         # Error log
#SBATCH --time=01:00:00           # Time limit hh:mm:ss (max 120:00:00)
#SBATCH --ntasks=1                # Number of tasks
#SBATCH --cpus-per-task=4         # CPU cores per task
#SBATCH --mem-per-cpu=1024                  # Memory per cpu

# Load modules and activate venv
module load python/3.8
source ~/venv/bin/activate

# Run your command here
python path/to/your/script.py
```
To create a shell file, just enter
```sh
$ vim script.sh
```
After writing the shell file, make it executable by typing
```sh
$ chmod +x script.sh
```
Finally, the job can be submitted by entering
```sh
$ sbatch script.sh
```
