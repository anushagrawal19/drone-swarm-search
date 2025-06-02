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

To run:

```bash
python3.11 ppo_trainer_base.py
```

### For Energy Environment:

* [ppo\_env\_energy.py](ppo_env_energy.py)
* [ppo\_trainer\_energy.py](ppo_trainer_energy.py)

To run:

```bash
python3.11 ppo_trainer_energy.py
```


## Acknowledgements

This project is based on the work from the [drone-swarm-search](https://github.com/pfeinsper/drone-swarm-search) repository.

## How to Cite

If you use this package, please consider citing it with the following BibTeX:

```
@software{Laffranchi_Falcao_DSSE_An_environment_2024,
    author = {
                Laffranchi Falcão, Renato and
                Custódio Campos de Oliveira, Jorás and
                Britto Aragão Andrade, Pedro Henrique and
                Ribeiro Rodrigues, Ricardo and
                Jailson Barth, Fabrício and
                Basso Brancalion, José Fernando
            },
    doi = {10.5281/zenodo.12659848},
    title = {{DSSE: An environment for simulation of reinforcement learning-empowered drone swarm maritime search and rescue missions}},
    url = {https://doi.org/10.5281/zenodo.12659848},
    version = {0.2.5},
    month = jul,
    year = {2024}
}
```






