# Distribution Generalization in Dynamical Real-World Systems
This git implements some pooling and domain generalization methods, applied to the FLUXNET data. The methods implemented are the following
1. Linear Regression (LR),
2. Extreme Gradient Boosting (XGBoost),
3. Long Short-Term Memory (LSTM),
4. Invariant Risk Minimization (IRM),
5. Stabilized Regression (SR).

To reproduce the experiments, ensure you have access to the FLUXNET data. Then, clone this repository into your preffered code editor and, in the main directory, create a folder called "Data". Next, upload the data into this folder. Due to the large amount of data, this can take up to an hour. Make sure that the "Data" folder is listed in .gitignore. This ensures that you don't upload large amount of data into Github when pushing changes to the repository.

## Structure of the git
This git implements the above listed methods. It assumes that data is uploaded into the main directory in a folder called "Data". For every method, there is a corresponding folder in the main directoy. It contains the three different Set-Ups **In-Site**, **LOSO** and **LOGO**. Every Set-Up contains the **python** file for the implementation of the method in the corresponding setting, a **shell** file for scheduling the jobs and a **csv**  file with the results. The use of the shell files is for scheduling the jobs as it is needed when working with Euler. However, if you are not working with a system requiring scheduling, I still recommend to look into the shell file to see how many ressources are necessary so that the job works. The ressources requested in the shell file are not optimal, in the sense that the code might also work with less ressources. However, requesting the same ressources will surely execute the job successfully.

## Guide on how to reproduce the experiments
After cloning the git and uploading the data, I give you now a short guide on how to reproduce experiments. For this reason, assume that we want to reproduce the experiments of the linear regression method in the In-Site setting.
The first thing you have to do is to create a virtual environment.

### Setting up a virtual environment
The cleanest way to be able to run the experiments is to use a virtual environment. A list of the dependencies needed in this environment for running the experiments can be found in the **requirements.txt** file. Use this file to create the virtual environment.

After creating the virtual environment, we are set to schedule the jobs.

### Scheduling a job
The first step in order to run the experiment is to change into the corresponding directory by typing "cd LR/In_Site/" into the terminal. After you changed the directory, you can submit the job by entering "sbatch script.sh" in the terminal. This will batch the shell file **script.sh**. Now, the job is already scheduled and depending on your priority in the queue you have to wait some time until the job is running. To check whether the job is already being executed or still queueing you can type "squeue -u <username>" into the terminal. If there are some intermediate prints within the python code, you can check them in the log-files which are being created for every job submitted.

Now, you know how to reproduce the experiments. Next, I want to explain how you run your own experiments on the FLUXNET data.

## Running your own methods
Assume you already uploaded the data and created a virtual environment. The first step is to consider the preprocessing of the data.

### Preprocessing data
The implementation of the preprocessing can be found in **preprocessing/preprocessing.py**. The file reads in the data and exectues the preprocessing steps as discussed in the thesis in Chapter 2.  If you are interested in predicting also other fluxes than the GPP, you can adjust the python file accordingly. For example if you want to consider the NEE as target variable, then include it in the **initialize_dataframe** function. Also don't forget to include the corresponding quality control variable NEE_qc, which will be used later when filtering for good quality data.

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
### Reading the data and defining features and target variable
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
Next, we for every site do a chronological 80/20 train/test split, then scale the data and finally train the model and do the predictions. Moreover, save the metrics you are interested in
```python
# Process each selected site
for site in sites_to_process:
    group = df[df['site_id'] == site]
    
    # Perform an 80/20 chronological split
    n_train = int(len(group) * 0.8)
    train = group.iloc[:n_train]
    test  = group.iloc[n_train:]
    
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
The LinearRegressor() can in principle be replaced by any regressor from sklearn. For pytorch based implementations, you can follow the In-Site code for the IRM and for tensorflow based implementations have a chek at the LSTM implementation. Given the pyhton file, you can run it using a shell file for scheduling. This already concludes how you can run the In-Site experiment with the method of your choice.
### Shell file
The shell files are important when you work on clusters which require scheduling jobs. In a shell file, you first specify some parameters like the time and memory you need for running the job. Moreover, you specify the modules you need to load so that the packages you use in the python scripts are loaded appropriately. Next to the modules, you will also activate the virtual environment which you created. In the end of the shell file, you express the command which you want to be executed, for example **python path/to/your/script.py**. A shell file can look like this
```sh
#!/bin/bash
# Sample shell script for SLURM job submission

#SBATCH --job-name=my_job         # Job name
#SBATCH --output=output.log       # Standard output log
#SBATCH --error=error.log         # Error log
#SBATCH --time=01:00:00           # Time limit hh:mm:ss
#SBATCH --ntasks=1                # Number of tasks
#SBATCH --cpus-per-task=4         # CPU cores per task
#SBATCH --mem=8G                  # Memory per node

# Load modules and activate venv
module load python/3.8
source ~/venv/bin/activate

# Run your command here
python path/to/your/script.py
```
To create a shell file, just type "vim script.sh" and then "i" to be able to modify the shell script. When the shell file is complete, click "Escape", then enter ":wq" and click "Enter". When the shell file was created for the first time, you need to make it executable by entering "chmod +x script.sh" in the terminal. Now, the job can be submitted by entering "sbatch script.sh" into the terminal.
