import pandas as pd
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from typing import Tuple
import os

def preprocess_house_data(
    input_path: str, 
    output_path: str,
    numerical_cols: list = ['Area', 'Bedrooms', 'Bathrooms', 'Floors', 'YearBuilt'],
    categorical_cols: list = ['Location', 'Condition', 'Garage']
) -> Tuple[pd.DataFrame]:
    processed_df = pd.read_csv(input_path)

    processed_df = processed_df.dropna().drop_duplicates()

    scaler = MinMaxScaler()

    processed_df[numerical_cols] = scaler.fit_transform(processed_df[numerical_cols])

    label_encoder = OneHotEncoder(sparse_output=False) # Set sparse_output to False to get a dense array
    encoded_features = label_encoder.fit_transform(processed_df[categorical_cols])

    # Create a DataFrame from the encoded features
    encoded_df = pd.DataFrame(encoded_features, columns=label_encoder.get_feature_names_out(categorical_cols))

    # Drop the original categorical features
    processed_df = processed_df.drop(categorical_cols, axis=1)

    # Concatenate the processed_df with the new encoded_df
    processed_df = pd.concat([processed_df, encoded_df], axis=1)

    processed_df.to_csv(output_path, index=False)

    return processed_df

# Example usage
if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.abspath(__file__))
    root_dir = os.path.dirname(script_dir)

    input_path = os.path.join(root_dir, 'house_dataset_raw.csv')
    output_path = os.path.join(script_dir, 'house_dataset_processed.csv')
    
    processed_df = preprocess_house_data(input_path, output_path)