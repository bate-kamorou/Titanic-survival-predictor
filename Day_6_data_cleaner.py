import pandas as pd 
from sklearn.preprocessing import MinMaxScaler
import joblib

class DataCleaner:
    """ 
    Takes in a messy dataset, clean it and return a cleaned dataset

    Methods:

    handel_missing_data(column, strategy=mean)
        fill in missing values of a  column based on the given strategy
    
    min_max_scale(column)
         Applies min max scaling to the specifed column 

    encode_categorical(column)
        One hot encode a list of specified columns 

    remove_columns(column)
        remove unneccessary  columns forn the data set
    
    get_clean_data()
        Return the clean dataset

    clean_all(missing_column, strategy, scale_column, encode_column, remove_column) 
        return the fully processed data 

    """

    def __init__(self, file_path=None) : 
        # read the file into a dataframe
        """
        Initializes with an optional file_path.

        if no path provided, self.df starts as None

        args:

        file_path: the path to the data to be cleaned

        """
        # initialize the sclaler
        self.scaler = MinMaxScaler()

        if file_path :
            self.df = pd.read_csv(file_path)
            print(f"Data loaded successfully. shape {self.df.shape}")
        else:
            self.df = pd.DataFrame()
            print("init erroring : No file path provided. Please load data using file_path in __init__")
            

    def handle_missing_data(self, column:str,strategy:str = "median") -> None:
        """
        fill in missing values for column based of a given strategy (mean, median, mode)

        args:

        column:  column with missing values to be filled
        strategy: how to fill in the missing values (mean, median, mode)
        """

        # fill in the missing values based on strategy 
        if self.df.empty :
            print("handel missing data erroing: Error No data loaded. Please load data first using file_path in __init__")
            return
        
        try :

            if strategy == "mean":
                col_mean = self.df[column].mean()
                print("column mean:",col_mean)
                self.df[column] = self.df[column].fillna(col_mean)
            elif strategy == "median":
                col_median = self.df[column].median()
                print("column median:", col_median)
                self.df[column] = self.df[column].fillna(col_median)
            elif strategy == "mode":
                col_mode = self.df[column].mode()
                print("column[] mode:", col_mode)
                self.df[column] = self.df[column].fillna(col_mode[0])
            else:
                print("strategy value not recognised use mean, median or mode")
        except ValueError as e :
            print(f"Error:  {e}")         

    def min_max_scale(self, column:str,is_training=True)-> None:
        """
        Applies min max scaling to the specifed column 

        args:

        column: the specified column to apply the scaling on 
        is_training: if True fit and transform the column with the scaler
                     if False only transform the column with the sclaer
        """
        
        if self.df.empty:
            print(" min max erroring: Error No data loaded. Please load data first using file_path in __init__")
            return
        
        try:
            if is_training:
                self.df[[column]] = self.scaler.fit_transform(self.df[[column]])
                print("column fitted and transformed during training")
            else:
                self.df[[column]] = self.scaler.transform(self.df[[column]])
                print("column transformed during inference")
        except Exception as e:
            print(f"Scaling error: {e}")
            
    def save_scaler(self, path:str):
        """
        save the fitted scaler object to a file

        args
            path: path  to save the scaler to 
        """
        joblib.dump(self.scaler, path)
        print(f"Scaler save to {path}")

    def load_scaler(self, path:str):
        """
        load the saved scaler object form a file

        args
            path : path to load the saved scaler from
        """
        self.scaler = joblib.load(path)
        print(f"Scaler loaded from {path}")

    def encode_categorical(self, columns:list) -> None:
        """
        One hot encode a list of specified columns 

        args:

        columns : the specified columns to one hot encode
        """
        if self.df.empty:
            print("encode categorical erroring: Error No data loaded. Please load data first using file_path in __init__")
            return
        
        try:
            # one hot encode the given columns
            self.df = pd.get_dummies(self.df, columns=columns, drop_first=True, dtype=int)
        except Exception as e:
            print(f"error: column name not recognized {e}")

    def remove_columns(self, columns:list):
        """ 
        remove unneccessary  columns forn the data set

        args:

        columns: names of columns to be removed from the dataset 
        """
        if self.df.empty:
            print("remove columns erroring: Error No data loaded. Please load data first using file_path in __init__")
            return
        
        # drop the columns
        self.df = self.df.drop(columns=columns)

    def get_clean_data(self) :
        """
        Return the clean dataset
        """
        # return the dataset
        return self.df 
    
    def clean_all(self,
                   missing_col:str,
                   strategy:str, 
                   scale_col:str, 
                   encode_cols:list[str],
                   remove_cols:list[str],
                   is_training:bool = True ):
        """
        Run the full titanic cleaning recipe in the correct order.

        Args: 
            missing_cols: the column with missing values to be filled 
            strategy: the strategy to fill the column with missing values
            encode_cols: categorical columns to be one hot encoded
            remove_cols: unnecessary columns to be removed from the data
        """
        # 1. fill in the missing  Age values 
        self.handle_missing_data(missing_col, strategy)

        # 2. Scale the Fare column
        self.min_max_scale(scale_col, is_training=is_training)

        # 3. one hot encode categorical    
        self.encode_categorical(encode_cols)

        # 4. remove unnecessary columns  
        self.remove_columns(remove_cols)

        print("full data cleaning process completed")

        return self.get_clean_data()






