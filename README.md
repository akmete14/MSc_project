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

## Guide how to run experiments
After cloning the git and uploading the data, I give you now a short guide on how to run the experiments.

### Setting up a virtual environment
The cleanest way to be able to run the experiments is to use a virtual environment. In order to run the experiments, you need to consider creating a virtual environment. A list of the dependencies needed in this environment for running the experiments can be found in the **requirements.txt** file. Use this file to create the virtual environment. Now that the data is loaded and the venv created, we can proceed to reproduce the experiments


### Creating a shell file
Assume that you have a python file **experiment.py** where you implement a method in a particular setting. Now, you want 

### Scheduling a job
Assume you want to run the linear regression model in the In-Site setting. Then, you first change into the corresponding directory by typing "cd LR/In_Site/" into the terminal. After you changed the directory, you can submit the job by entering "sbatch script.sh" in the bash. This will batch the shell file **script.sh**. 
This will schedule the job and depending on your priority in the queue you wait until the job is running. You can backtrack whether the code is working by looking at the log-files which are created as soon as the job is running.


## Running your own methods
Explain how to run same experiments when trying another method
