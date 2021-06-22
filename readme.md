### Prerequisites
For running the `test.py` file you need to use python 3.7 or higher

### Folder structure

The project is divided in the following packages:
* **scripts/autog/**: Contains all packages required for our auto-grad framework
* **test.py**: This file trains and tests, one model for 10 runs, and prints for each:
  * Error rate in each run
  * Mean error rate and std for 10 runs

### Run
To run simply navigate to the folder root and use:
```bash
$> python test.py
```
