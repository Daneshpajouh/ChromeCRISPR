# Navitas
Course project for directed research course

## Data Source
Störtz, F. and Minary, P. (2021).  crisprSQL: a novel database platform for crispr/casoff-target cleavage assays._Nucleic Acids Research_,**49**(D1), D855–D861

## Dependencies

* Pandas: 
    * pandas development team, T. (2020). pandas-dev/pandas: Pandas.
    * Wes McKinney (2010). Data Structures for Statistical Computing in Python. In Stéfan van der Walt and Jarrod Millman, editors, Proceedings of the 9th Python in Science Conference, pages 56 – 61.
* Numpy: 
   *  Harris, C. R. et al. (2020). Array programming with NumPy. Nature, 585(7825), 357–362.
    
* Pytorch:
    * Paszke, A. et al. (2019). Pytorch: An imperative style, high-performance deep learning library. In H. Wallach, H. Larochelle, A. Beygelzimer, F. d'Alché-Buc, E. Fox, and R. Garnett, editors, Advances in Neural Information Processing Systems 32 , pages 8024–8035. Curran Associates, Inc.
* sklearn:
    * Pedregosa, F. et al. (2011). Scikit-learn: Machine learning in Python. Journal of Machine Learning Research, 12, 2825–2830.
* scipy:
    * Virtanen, P. et al. (2020). SciPy 1.0: Fundamental Algorithms for Scientific Computing in Python. Nature Methods, 17, 261–272.
* itertools:
    * Van Rossum, G. (2020). The Python Library Reference, release 3.8.2 . Python Software Foundation.
* pickle:
    * Van Rossum, G. (2020). The Python Library Reference, release 3.8.2 . Python Software Foundation.
* functools:
    * Van Rossum, G. (2020). The Python Library Reference, release 3.8.2 . Python Software Foundation.

## Sample Session

### Configuration
Go to `src/config.py`
Set `MODEL` equal to `models.LinearRegressor` or `models.LinearRegressor2`
Set the `MODEL_FILE_PATH` to anything with a `.pkl` extension
Set the `HYPERPARAMS_FILE_PATH` and `PREDICTIONS_FILE_PATH` to anything with a `.csv` extension

By default, the dataframes used are from a previously split crisprSQL dataset. This can be changed by changing the `df` for new training data or the `df_test` for the testing data. The split size of the training data and validation data can also be altered by changing `test_size`.

If you are in the validation stage of the model, set `df_predict = df_validate`. Otherwise if you are in the test stage, set `df_predict = df_test`.
```
# Select Model and File Path
MODEL = models.LinearRegressor2
MODEL_FILE_PATH = "./src/model2.pkl"
HYPERPARAMS_FILE_PATH = "./results/hyperparameters/model2.csv"
PREDICTIONS_FILE_PATH = "./results/predictions/model2.csv"

# Get dataframes
df = pd.read_csv("./data/train.csv")
df_train, df_validate = train_test_split(df, test_size = 0.2)
df_test = pd.read_csv("./data/test.csv")

# Predict validation or test set?
df_predict = df_validate
```

### Tune Hyperparameters
Go ro `run.sh` and comment out the rest of the code:
```
echo Tuning Hyperparameters...
./src/tune_hyperparameters.py
echo Done.
echo
echo
#echo Training...
#./src/train.py
#echo Done.
#echo
#echo
#echo Predicting...
#./src/predict.py
#echo
#echo
echo Have a nice day
```
To test different epochs, batch sizes, and learning rates, go to `src/tune_hyperparameters.py`.
To get a csv output, use the `tune_hyperparameters` function. To plot the loss over each epoch, use the `plot_loss` function. 
The csv file will be output to whatever was set as the `HYPERPARAMS_FILE_PATH` in `src/config.py`. The loss plots will be in `results/tuning`.

### Training the Model
Similarly to tuning the hyperparameters, go to `run.sh` and comment out all code except `./src/train.py`. 

### Making Predictions
Similarly to tuning the hyperparameters, go to `run.sh` and comment out all code except `./src/predict.py`. Ensure you have set the correct dataset in `src/config.py` for validating or testing. 

### The Models
The parameters to be used from the data set are defined as class variables. 
```
class LinearRegressor:
    def __init__(self, model_file_path):
        self.model_file_path = model_file_path
        self.params = ["grna_target_sequence", "target_sequence"]
```
The train function drops any rows in th dataset that contain a value of NaN in either of the sequence columns or the cleavage frequency column. It then gets tensors `X` and `y` from the `Preprocessing.get_X` and `Preprocssing.get_y` functions.
```
def train(self, df_train):
        df_train = Preprocessing.drop_na(df_train, self.params)
        X = Preprocessing.get_X(df_train, self.params)
        y = Preprocessing.get_y(df_train)
```
The model type, loss function, optimization function, and hyperparameters are set manually, based on the outcome of hyperparameter tuning:
```
        model = nn.Linear(input_dim, 1)
        loss_func = nn.MSELoss()
        optimizer = torch.optim.SGD(model.parameters(), lr = 0.0001)
        BATCH_SIZE = 1
        EPOCH = 25
```
The model is then trained by the following piece of code:
```
 for epoch in range(EPOCH):
            epoch_loss = []
            for step, (batch_x, batch_y) in enumerate(loader):
                prediction = model(batch_x)
                loss = loss_func(prediction, batch_y)
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
```
and saved to as a pickle file
```
with open(self.model_file_path, 'wb') as model_file:
            pickle.dump(model, model_file)
```
The train function also produces a loss curve saved in results. 

### Preprocessing Data
The `src/transformations.py` file contains all the preprocessing done to the data. Of note, the `LinearRegressor` model's sequences are encoded using sequential encoding:
```
    def encode_nt(nt:str) -> int:
        assert len(nt) == 1
        encoding_dict = {
                'X': 0,
                'A': 0.25,
                'T': 0.5,
                'G': 0.75,
                'C': 1
        }
        return encoding_dict.get(nt.upper())
```

and the `LinearRegressor2` model has it's sequences encoded by the one-hot encoding function:
```  
    def encode_nt_onehot(nt:str) -> int:
        assert len(nt) == 1
        encoding_dict = {
            'X': [0, 0, 0, 0],
            'A': [1, 0, 0, 0],
            'T': [0, 1, 0, 0],
            'G': [0, 0, 1, 0],
            'C': [0, 0, 0, 1]
        }
        return encoding_dict.get(nt.upper())
```

## Listing

### tune_hyperparameters.py
##### benchmark_hyperparams
Input: Cleavage rate predictions as a Numpy array, True cleavage rates as a numpy array
Output: mean squared error, Spearman correlation, Spearman pvalue  
Description: Calculates some benchmark statistics on cleavage rate predictions

##### tune_hyperparams
Input: Training dataset
Output: csv file containing mean squared error and Spearman correlation for each set of hyperparameters tested. 
Description: Generates a csv file containing benchmark statistics for the hyperparameters epoch, batch, and learning rate as described in the function

#### plot_loss
Input: Training dataset
Output: png files showing the loss over each epoch
Description: Loss plots files are stored in /results/tuning. 

### train.py
#### train
Input: None
Output: None
Description: Creates a pickled version of the model at a predetermined file path

### predict.py
#### predict
Input: Dataset to be predicted
Output: Predictions, Actual cleavage rates
Description: Uses a pretrained model saved at a predetermined file path to make cleavage rate predictions on a give dataset

#### benchmark
Input: Predictions, Actual cleavage rates
Output: Display on screen
Description: Prints mean squared error, Spearman correlation, and Spearman pvalue to screen

### models.py
#### train
Input: Training dataframe
Output: None
Description: Trains a model using manually entered hyperparameters and saves it as a pickle file

#### predict
Input: Dataset to be predicted
Output: Predictions, Actual cleavage rates, Dataset used for prediciton
Description: Uses a pretrained model saved at a predetermined file path to make cleavage rate predictions on a give dataset

#### train_hyperparams
Input: Training dataset, batch, epoch, learning rate)
Output: None
Description: Creates a pickled model using the input hyperparameters

#### tune_hyperparameters
Input: Training dataset, learning rate
Output: png file with loss over each epoch
Description: Loss plots files are stored in /results/tuning. 

### transformations.py
