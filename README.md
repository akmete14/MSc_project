# Distribution Generalization in Dynamical Real-World Systems
This git implements some pooling and domain generalization methods, applied to the FLUXNET data. The methods implemented are the following
1. Linear Regression (LR),
2. Extreme Gradient Boosting (XGBoost),
3. Long Short-Term Memory (LSTM),
4. Invariant Risk Minimization (IRM),
5. Stabilized Regression (SR).

To reproduce the experiments, ensure you have access to the FLUXNET data. Then, clone this repository into your preffered code editor and, in the main directory, create a folder called "Data". Next, upload the data into this folder. Due to the large amount of data, this can take up to an hour. Make sure that the "Data" folder is listed in .gitignore. This ensures that you don't upload large amount of data into Github when pushing changes to the repository.

## Structure of the git
This git implements the above listed methods. It assumes that data is uploaded into the main directory in a folder called "Data". For every method, there is corresponding folder in the main directoy. It contains the three different Set-Ups **In-Site**, **LOSO** and **LOGO**. Every Set-Up contains the **python** file for the implementation of the method in the corresponding setting, a **shell** file for scheduling the jobs and a **csv**  file with the results. The use of the shell files is for scheduling the jobs as it is needed when working with Euler. However, if you are not working with scheduling, it still makes sense to look into the shell file to see how many ressources are necessary so that the job works. The ressources requested in the shell file are not optimal, in the sense that the code might also work with less ressources. However, requesting the same ressources will surely finish the job successfully.

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
Assume you already uploaded the data and created a virtual environment.

### Preprocessing data
The implementation of the preprocessing can be found in **../preprocessing/preprocessing.py**. The file reads in the data and exectues the preprocessing steps as discussed in the thesis in Chapter 2.  If you are interested in predicting also other fluxes than the GPP you can adjust the python file accordingly. If, for example, you want to consider the NEE as target variable, then include it in the **initialize_dataframe** function. Also don't forget to include the corresponding quality control variable NEE_qc, which will be used later for filtering for good quality data.

