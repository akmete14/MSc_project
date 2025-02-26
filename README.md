# Distribution Generalization in Dynamical Real-World Systems
This git implements some pooling and domain generalization methods, applied to the FLUXNET data. The methods implemented are the following
1. Linear Regression (LR),
2. Extreme Gradient Boosting (XGBoost),
3. Long Short-Term Memory (LSTM),
4. Invariant Risk Minimization (IRM),
5. Stabilized Regression (SR).

To reproduce the experiments, ensure you have access to the FLUXNET data. Next, clone this repository. In the main directory, create a folder called "Data" and copy the data into this folder. Due to the large amount of data, this can take a while. Make sure that the "Data" folder is listed in **.gitignore** to ensure that you don't upload large amounts of data when pushing changes to the repository.

## Structure of the git
This git implements the above listed methods. It assumes that data is copied into the main directory in a folder called "Data". For every method, there is a corresponding folder in the main directoy. It contains three different Set-Ups **In-Site**, **LOSO** and **LOGO**. Every Set-Up contains a **python** file for the implementation of the method in the corresponding setting, a **shell** file for scheduling the jobs and a **csv**  file with the results. The use of the shell files is for scheduling the jobs as it is needed when working with Euler. However, if you are not working with a system requiring scheduling, I still recommend to look into the shell file to see how many ressources are necessary so that the job works. The ressources requested in the shell file are not optimal, in the sense that the code might also work with less ressources. However, requesting the same ressources will surely execute the job successfully.

## Guide on how to reproduce the experiments
### Setting up a virtual environment
The cleanest way to be able to run the experiments is by creating a virtual environment. A list of the dependencies needed in this environment can be found in the **requirements.txt** file. Use this file to create the virtual environment.
After creating the virtual environment, we are set to reproduce the experiments.

As an example to show how to schedule jobs, we will schedule the job for the In-Site experiment of the Linear Regression method.
### Scheduling a job
The first step in order to run the experiment is to change into the corresponding directory, in our case by typing
```sh
$ cd LR/In_Site/
```
in the terminal. After changing the directory, you can submit the job by entering 
```sh
$ sbatch script.sh
```
in the terminal. This will batch the shell file **script.sh**. If the job was successfully submitted, you should see an output like
```sh
$ sbatch script.sh
Submitted batch job 1234567
```
The job is now scheduled and depending on your priority in the queue you have to wait some time until the job is being executed. To check the status of the job, type in 
```sh
$ squeue -u<username>
```
in the terminal (replace <username> with your username). If there are some intermediate prints within the python code, you can check them in the log-files which are being created as soon as the job is being executed. Now, you know how to reproduce all experiments which are already implemented. Next, I explain how you can run your own experiments on the FLUXNET data. I will show it at the example of the In-Site experimental setting.

## Running your own methods
Assume you already uploaded the data and created a virtual environment. The first step is to consider the preprocessing of the data.

### Preprocessing data
The implementation of the preprocessing can be found in **preprocessing/preprocessing.py**. The file reads in the data and exectues the preprocessing steps as discussed in the thesis in Chapter 2.  If you are interested in predicting also other fluxes next to the GPP, you can adjust the python file accordingly. For example if you want to consider the NEE as target variable, then include it in the **initialize_dataframe** function. Also don't forget to include the corresponding quality control variable NEE_qc, which will be used later when filtering for good quality data.

Below is a snippet from `../preprocessing/preprocessing.py`, but including the NEE:

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
Given the desired preprocessed data, you can start setting up the In-Site experiment. Doing so depends highly on the implementation of the method. Generally, the structure is as follows
### Reading the data and defining features & target variable
First, read in the data which we preprocessed and define the feature set and target variable:
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
Given all sitenames (from the site_id column of the dataframe from the preprocessing), we can consider now the for loop. That is, for every site we do the following: First we split the data of the site into a chronological 80/20 train/test split
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

````
Finally, we define the regressor, fit the model and evaluate using different metrics
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
The LinearRegressor() can in principle be replaced by any regressor from sklearn. For pytorch based implementations, you can follow the In-Site code for the IRM and for tensorflow based implementations chek the LSTM implementation. Given the pyhton file, you can run it using a shell file for scheduling.
### Shell file
The shell files are important when you work on clusters which require scheduling jobs. In a shell file, you first specify some parameters like time and memory you need for running the job. Moreover, you specify the modules you need to load so that the packages you import in the python scripts are loaded appropriately. Next to the modules, you also activate the virtual environment which you created. In the end of the shell file, you express the command which you want to be executed, for example **python path/to/your/script.py**. A shell file can look like this
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
Now, the job can be submitted by entering
```sh
$ sbatch script.sh
```
