## Acknowledgements

This project is based on the work from the [drone-swarm-search](https://github.com/pfeinsper/drone-swarm-search) repository.

# Installation and Setup Guide

Follow these steps to install and run the project.

## Step 1: Install Python 3.11

### Using Homebrew (macOS/Linux)

If you're on macOS or Linux and don't have Python 3.11 installed, you can use **Homebrew**:

```
brew install python@3.11
```

### Via Python.org

Go to official python website to download and install Python 3.11


## Step 2: Verify Python Version

To ensure you're using Python 3.11 in your virtual environment:

```bash
python --version
```

You should see something like:

```bash
Python 3.11.x
```

If it shows a different version, change to use python 3.11


## Step 3: Set Up a Virtual Environment

1. **`cd` to the project folder** in the terminal.

2. Create a new **virtual environment** using Python 3.11:

    ```
    python3.11 -m venv .venv
    ```

3. **Activate the virtual environment**:

    ```
    source .venv/bin/activate
    ```

   Once activated, your terminal should show the virtual environment name `.venv` in the prompt (e.g., `(.venv)`).


## Step 4: Install Project Dependencies

Now that your virtual environment is set up, install all necessary dependencies by running:

```
pip3.11 install -r requirements.txt
```

This will install all the packages listed in `requirements.txt` that are required for the project.

## Step 5: Run the Project

Now you can run the project.

The following are the main training files:

### For Base Environment:

* [ppo\_env\_base.py](ppo_env_base.py)
* [ppo\_trainer\_base.py](ppo_trainer_base.py)


### For Energy Environment:

* [ppo\_env\_energy.py](ppo_env_energy.py)
* [ppo\_trainer\_energy.py](ppo_trainer_energy.py)


### Command-line Arguments:

The following arguments can be used when running the training scripts:

* `--mode [train|test]`: Specifies whether to run in **train** or **test** mode.

  **Default**: `train`

* `--model_path [logs/base/best_model.pt | logs/energy/best_model.pt]`: Specifies the path to the model checkpoint file.
  This needs to be specified **only in test mode**.

* `--episodes [num]`: Specifies the number of episodes to run.

  **Default**:
  * `train`: 5000 episodes
  * `test`: 100 episodes

* `--render_mode [ansi|human]`: Specifies the rendering mode for the environment.

  * `ansi`: No graphics.
  * `human`: Displays a graphical interface.

    **Default**: `ansi`

### Example Usage:

To train the Energy Environment:

```bash
python3.11 ppo_trainer_energy.py --mode train --episodes 5000 --render_mode ansi
```

To test the Energy Environment with a trained model:

```bash
python3.11 ppo_trainer_energy.py --mode test --model_path logs/energy/best_model.pt --episodes 100 --render_mode ansi
```

To train the Base Environment:

```bash
python3.11 ppo_trainer_base.py --mode train --episodes 5000 --render_mode ansi
```

To test the Base Environment with a trained model:

```bash
python3.11 ppo_trainer_base.py --mode test --model_path logs/base/best_model.pt --episodes 100 --render_mode ansi
```

### Modify FPS in Human Mode:

To change the FPS when in **human mode**, go to [DSSE/environment/pygame\_interface.py](DSSE/environment/pygame_interface.py) and modify the `FPS` variable.





## Important Directories and Files

* **[logs](logs) directory**: This is where all the training logs and models are stored.

* **[test\_logs](test_logs) directory**: This is where all the test results are stored.

* **[DSSE](DSSE) directory**: This is the main environment directory that contains the entities, simulation, environment logic, interface, etc.

### Key Files in the Project:

* **[ppo\_env\_base.py](ppo_env_base.py)**: Environment geared towards **PPO** (Proximal Policy Optimization) training.

* **[ppo\_trainer\_base.py](ppo_trainer_base.py)**: Training and testing file for the base **PPO** model.

* **[ppo\_env\_energy.py](ppo_env_energy.py)**: Environment geared towards **PPO** training, with energy logic included.

* **[ppo\_trainer\_energy.py](ppo_trainer_energy.py)**: Training and testing file for **PPO** with energy-aware logic.

---