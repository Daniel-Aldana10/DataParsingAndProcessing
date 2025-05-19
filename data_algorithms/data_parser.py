import pandas as pd
import numpy as np
from pathlib import Path
import os
import json
import pickle


class EmployeeDataParser:
   
    
    def __init__(self, file_path=None):
        
        self.file_path = file_path
        self.data = None
        self.original_data = None
    
    def load_data(self, file_path=None):
       
        if file_path:
            self.file_path = file_path
        
        if not self.file_path:
            raise ValueError("No file path provided")
        
        file_path_obj = Path(self.file_path)
        if not file_path_obj.exists():
            raise FileNotFoundError(f"File {self.file_path} does not exist")
        
        self.data = pd.read_csv(self.file_path)
        # Save copy of original data
        self.original_data = self.data.copy()
        return self.data
    
    def clean_data(self):
        if self.data is None:
            raise ValueError("No data has been loaded. Use load_data() first.")
        
        # Make a copy to avoid modifying original data
        df = self.data.copy()
        
        # Convert dates to datetime format
        date_columns = ['hire_date', 'last_review_date', 'birth_date']
        for col in date_columns:
            if col in df.columns:
                df[col] = pd.to_datetime(df[col], errors='coerce')
        
        # Convert numeric values
        numeric_columns = ['days_service', 'base_salary', 'bonus_percentage', 
                          'performance_score', 'vacation_days', 'sick_days']
        for col in numeric_columns:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # Remove completely empty rows
        df = df.dropna(how='all')
        
        # Fill missing values in categorical columns with 'Unknown'
        categorical_columns = ['department', 'job_title', 'status', 'city', 
                              'state', 'country', 'gender', 'education', 'work_location', 'shift']
        for col in categorical_columns:
            if col in df.columns:
                df[col] = df[col].fillna('Unknown')
        
        # Create useful derived columns
        if 'birth_date' in df.columns:
            current_year = pd.Timestamp.now().year
            df['age'] = current_year - df['birth_date'].dt.year
        
        if 'base_salary' in df.columns and 'bonus_percentage' in df.columns:
            df['total_compensation'] = df['base_salary'] * (1 + df['bonus_percentage'] / 100)
        
        # New derived columns
        if 'hire_date' in df.columns:
            df['tenure_years'] = (pd.Timestamp.now() - df['hire_date']).dt.days / 365.25
            # Categorize tenure
            df['tenure_category'] = pd.cut(
                df['tenure_years'], 
                bins=[0, 1, 3, 5, 10, 100], 
                labels=['<1 year', '1-3 years', '3-5 years', '5-10 years', '>10 years']
            )
        
        # Salary/performance ratio if both columns exist
        if 'base_salary' in df.columns and 'performance_score' in df.columns:
            df['salary_per_performance'] = df['base_salary'] / df['performance_score']
        
        # Calculate age at hire
        if 'birth_date' in df.columns and 'hire_date' in df.columns:
            df['age_at_hire'] = (df['hire_date'] - df['birth_date']).dt.days / 365.25
        
        self.data = df
        return df
    
    def handle_duplicates(self, strategy='remove', subset=None):
        
        if self.data is None:
            raise ValueError("No data has been loaded. Use load_data() first.")
        
        if subset is None:
            subset = ['employee_id'] if 'employee_id' in self.data.columns else None
        
        # Detect duplicates
        duplicates = self.data.duplicated(subset=subset, keep=False)
        duplicate_count = duplicates.sum()
        
        print(f"Found {duplicate_count} duplicate rows.")
        
        if duplicate_count == 0:
            return self.data
        
        if strategy == 'remove':
            # Remove duplicates keeping first record
            self.data = self.data.drop_duplicates(subset=subset, keep='first')
        elif strategy in ['mean', 'max', 'min']:
            # Combine duplicates according to strategy
            numeric_cols = self.data.select_dtypes(include=['number']).columns.tolist()
            
            # Group by identification columns
            grouped = self.data.groupby(subset)
            
            if strategy == 'mean':
                aggregated = grouped[numeric_cols].mean()
            elif strategy == 'max':
                aggregated = grouped[numeric_cols].max()
            else:  # min
                aggregated = grouped[numeric_cols].min()
            
            # For categorical columns, take most frequent value
            categorical_cols = self.data.select_dtypes(include=['object']).columns.tolist()
            categorical_cols = [c for c in categorical_cols if c not in subset]
            
            if categorical_cols:
                # Function to get most frequent value
                def most_common(series):
                    return series.value_counts().index[0]
                
                cat_aggregated = grouped[categorical_cols].agg(most_common)
                
                # Combine numeric and categorical results
                aggregated = pd.concat([aggregated, cat_aggregated], axis=1)
            
            self.data = aggregated.reset_index()
        
        return self.data
        
    def detect_outliers(self, columns=None, method='zscore', threshold=3):
        if self.data is None:
            raise ValueError("No data has been loaded. Use load_data() first.")
        
        # If no columns specified, use all numeric ones
        if columns is None:
            columns = self.data.select_dtypes(include=['number']).columns.tolist()
        
        # Filter only numeric columns from provided list
        numeric_columns = [col for col in columns if col in self.data.select_dtypes(include=['number']).columns]
        
        if not numeric_columns:
            raise ValueError("No numeric columns found to analyze")
        
        outlier_mask = pd.Series(False, index=self.data.index)
        
        for col in numeric_columns:
            if method == 'zscore':
                # Z-score method
                zscore = np.abs((self.data[col] - self.data[col].mean()) / self.data[col].std())
                col_outliers = zscore > threshold
            else:  # IQR method
                # Interquartile range method
                Q1 = self.data[col].quantile(0.25)
                Q3 = self.data[col].quantile(0.75)
                IQR = Q3 - Q1
                col_outliers = (self.data[col] < (Q1 - threshold * IQR)) | (self.data[col] > (Q3 + threshold * IQR))
            
            # Update outliers mask
            outlier_mask = outlier_mask | col_outliers
        
        # Create DataFrame with outlier information
        outlier_info = {}
        for col in numeric_columns:
            if method == 'zscore':
                zscore = np.abs((self.data[col] - self.data[col].mean()) / self.data[col].std())
                outlier_info[f"{col}_zscore"] = zscore
                outlier_info[f"{col}_is_outlier"] = zscore > threshold
            else:
                Q1 = self.data[col].quantile(0.25)
                Q3 = self.data[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - threshold * IQR
                upper_bound = Q3 + threshold * IQR
                outlier_info[f"{col}_lower_bound"] = pd.Series([lower_bound] * len(self.data))
                outlier_info[f"{col}_upper_bound"] = pd.Series([upper_bound] * len(self.data))
                outlier_info[f"{col}_is_outlier"] = (self.data[col] < lower_bound) | (self.data[col] > upper_bound)
        
        outlier_df = pd.concat([self.data, pd.DataFrame(outlier_info)], axis=1)
        
        # Return only rows with outliers
        return outlier_df[outlier_mask]
    
    def normalize_categories(self, column, mapping=None, case_sensitive=False):
        if self.data is None:
            raise ValueError("No data has been loaded. Use load_data() first.")
            
        if column not in self.data.columns:
            raise ValueError(f"Column '{column}' does not exist in the data")
        
        # Create a copy of DataFrame to avoid modifying original
        df = self.data.copy()
        
        if mapping is None:
            # Get unique values
            unique_values = df[column].dropna().unique()
            
            # If not case sensitive, convert everything to lowercase
            if not case_sensitive:
                # Create dictionary of original values to lowercase
                lower_mapping = {}
                for val in unique_values:
                    lower_val = str(val).lower()
                    if lower_val not in lower_mapping:
                        lower_mapping[lower_val] = val
                    else:
                        # If multiple values convert to same lowercase,
                        # choose most common in data
                        current = lower_mapping[lower_val]
                        if df[column].value_counts()[val] > df[column].value_counts()[current]:
                            lower_mapping[lower_val] = val
                
                # Create mapping from original to normalized
                mapping = {val: lower_mapping[str(val).lower()] for val in unique_values}
        
        # Apply mapping
        df[column] = df[column].map(mapping).fillna(df[column])
        
        self.data = df
        return df
    
    def export_data(self, output_path, format='csv', index=False):
        if self.data is None:
            raise ValueError("No data has been loaded. Use load_data() first.")
        
        # Ensure directory exists
        output_dir = os.path.dirname(output_path)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        # Export according to requested format
        if format.lower() == 'csv':
            self.data.to_csv(f"{output_path}.csv", index=index)
        elif format.lower() == 'json':
            self.data.to_json(f"{output_path}.json", orient='records')
        elif format.lower() == 'excel':
            self.data.to_excel(f"{output_path}.xlsx", index=index)
        elif format.lower() == 'pickle':
            with open(f"{output_path}.pkl", 'wb') as f:
                pickle.dump(self.data, f)
        else:
            raise ValueError(f"Format '{format}' not supported. Use 'csv', 'json', 'excel' or 'pickle'")
        
        print(f"Data exported to {output_path}.{format.lower()}")
    
    def create_derived_variables(self):
        if self.data is None:
            raise ValueError("No data has been loaded. Use load_data() first.")
        
        df = self.data.copy()
        
        # Performance categories
        if 'performance_score' in df.columns:
            df['performance_category'] = pd.cut(
                df['performance_score'],
                bins=[0, 2, 3, 4, 5],
                labels=['Low', 'Medium', 'High', 'Excellent']
            )
        
        # Salary categories
        if 'base_salary' in df.columns:
            # Create salary quartiles
            df['salary_quartile'] = pd.qcut(
                df['base_salary'],
                q=4,
                labels=['Q1', 'Q2', 'Q3', 'Q4']
            )
        
        # Bonus ratio to base salary
        if 'base_salary' in df.columns and 'bonus_percentage' in df.columns:
            df['bonus_ratio'] = df['bonus_percentage'] / 100
        
        # Total compensation vs performance ratio
        if 'total_compensation' in df.columns and 'performance_score' in df.columns:
            df['comp_perf_ratio'] = df['total_compensation'] / df['performance_score']
        
        # Variable to identify high-value employees (high performance and low compensation)
        if 'performance_score' in df.columns and 'total_compensation' in df.columns:
            high_perf = df['performance_score'] > df['performance_score'].quantile(0.75)
            low_comp = df['total_compensation'] < df['total_compensation'].quantile(0.5)
            df['high_value_employee'] = high_perf & low_comp
        
        # Variable to identify turnover risk (based on performance and service time)
        if 'performance_score' in df.columns and 'days_service' in df.columns:
            low_perf = df['performance_score'] < df['performance_score'].quantile(0.25)
            medium_tenure = (df['days_service'] > 365) & (df['days_service'] < 3*365)
            df['turnover_risk'] = low_perf & medium_tenure
        
        # Days since last review
        if 'last_review_date' in df.columns:
            df['days_since_last_review'] = (pd.Timestamp.now() - df['last_review_date']).dt.days
        
        self.data = df
        return df
    
    def get_data(self):
        if self.data is None:
            raise ValueError("No data has been loaded. Use load_data() first.")
        return self.data
