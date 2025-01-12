# Timber! Poisoning decision trees

This repository contains the code implementation for the paper [Timber! Poisoning Decision Trees](https://arxiv.org/abs/2410.00862)' experiments.

## Setup
To execute the code, it is recommended to create a virtual environment or a conda environment. This is because we use a modified version of [SciKit-Learn](https://scikit-learn.org/stable/index.html) that requires compilation prior to usage.

#### Conda environment
To create a Conda environment, it is highly suggested to follow the [guide](https://scikit-learn.org/stable/developers/advanced_installation.html#building-from-source) provided by [SciKit-Learn](https://scikit-learn.org/stable/index.html), as it is the most reliable source for creating a functional environment to compile the library. Here, we outline the steps provided in the [guide](https://scikit-learn.org/stable/developers/advanced_installation.html#building-from-source), followed by the installation of some additional modules:

```sh
conda create -n timber-env -c conda-forge python numpy scipy cython meson-python ninja
```

```sh
conda activate timber-env
```

```sh
pip install matplotlib pandas attrs tqdm threadpoolctl colorama
```

#### Virtual Environment
As an alternative, you can create a virtual environment to compile the source code of the [SciKit-Learn](https://scikit-learn.org/stable/index.html) library locally. As previously mentioned, the most reliable method is to follow to the [guide](https://scikit-learn.org/stable/developers/advanced_installation.html#building-from-source) provided by [SciKit-Learn](https://scikit-learn.org/stable/index.html), followed by the installation of additional modules:

```sh
python3 -m venv timber-env
```

```sh
source timber-env/bin/activate
```

```sh
pip install wheel numpy scipy cython meson-python ninja
```

```sh
pip install matplotlib pandas attrs tqdm threadpoolctl colorama
```

### Compile the SciKit-Learn modified source
To execute the code, you must then compile the modified source code of [SciKit-Learn](https://scikit-learn.org/stable/index.html). This can be accomplished using the following command, as provided in the development [guide](https://scikit-learn.org/stable/developers/advanced_installation.html#building-from-source) for [SciKit-Learn](https://scikit-learn.org/stable/index.html):

```
pip install --editable . --verbose --no-build-isolation --config-settings editable-verbose=true
```

## Running the Code
To ensure reliable execution, we provide a JSON configuration file that defines an attack pipeline, enabling the execution of multiple attacks with identical parameters.

### Configuration of the pipeline.json file
The configuration file contains a default hyperparameter grid for a random forest model (the same grid used for the experiments conducted for the paper).

It also lists the datasets, configured as follows:

```json
“datasets”: [
    {“id”:”DATASET_ID”, “classes”: [POSITIVE_CLASS, NEGATIVE_CLASS]},
    …
]
```

**DATASET_ID** corresponds to one of the following options:
1. `breast_cancer` (for the Breast Cancer dataset)
2. `spam` (for the Spambase dataset)
3. `musk2` (for the Musk2 dataset)
4. `winecolor` (for the Wine dataset)

**POSITIVE_CLASS** and **NEGATIVE_CLASS** correspond to the values to use for the respective classes. By default, the configuration is `[1, 0]`, where `1` represents the positive class and `0` the negative class. Each attack will target the positive class; therefore, to attack the negative class, it is possible to set the classes to `[0, 1]`.

Finally, the last configuration node defines the attack pipeline itself.

```json
“attack_pipeline”: {
    “budget”: 0.1,
    “test_size”: 0.2,
    “random_state”: null,
    “decorate_syntheses”: true,
    “summarize”: true,
    “verbose”: true,
    “methods”: [“ges”, “tes”, “greedy”, “timber”, “random”]
}
```
**random_state** defines the random state set for the dataset split and model training.

If **decorate_syntheses** is set to `true`, the run will generate an additional annotated.csv file for each synthesis, which will contain the data collected from the attack and the model’s score for each iteration.

If **summarize** is set to `true` and **decorate_synthesis** is `true`, a 10 datapoint summary of the attack, along with the annotated iterations and the raw data collected from the attack, will also be generated.

### Running the code
Once the JSON configuration file is set up, the code can be run with

```sh
python run.py
```

The results will be stored in the `.synthesis/` directory.
The code will also generate some logs that can be inspected for additional information about the iterations (for greedy-like algorithms).

### Running the defence code
As for the attack code, the defence code configuration is set in the `pipeline.json` file. The configurated defences will run for all syntheses that have been generated previously in the attack phase.

The defence configuration node is set as follows:
```json
"defences": [
    {
        "kind": "<DEFENCE_NAME>",
        "param_grid": {
            "<PARAM_NAME>": [<PARAM_VALUES>],
            ...
        }
    }
]
```

- **DEFENCE_NAME** corresponds to the name of the defence to be used. Both implemented defences are avaliable and pre-configured in the provided `pipeline.json` file.
- **PARAM_NAME** corresponds to the name of the parameter to be tuned for the defence.
- **PARAM_VALUES** corresponds to the values to be tested for the parameter.

To run the defence code, execute the following command:

```sh
python defend.py
```

This will run the defences for every possible combination of hyperparameters defined in the `pipeline.json` file and store the results into properly named csv files inside the syntheses folders.