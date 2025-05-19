import pytest
import pandas as pd
import os
from data_algorithms.data_parser import EmployeeDataParser
from pathlib import Path


@pytest.fixture
def sample_csv_file(tmp_path):
    """Creates a temporary test CSV file."""
    file_path = tmp_path / "test_employee_data.csv"
    data = """employee_id,first_name,last_name,email,phone_number,department,job_title,hire_date,days_service,base_salary,bonus_percentage,status,birth_date,address,city,state,zip_code,country,gender,education,performance_score,last_review_date,employee_level,vacation_days,sick_days,work_location,shift,emergency_contact,ssn,bank_account
EMP3479083960,Jack,Blair,jack.blair@company.com,241.926.2177x652,Engineering,Mechanical Engineer,2022-05-12,1065,65242.57,1.76,Active,1972-02-17,1963 Miles River Suite 288,Columbus,OH,40450,USA,M,Professional,72.11,2024-01-31,Senior,13,5,Remote,Day,(406)871-7253,547-45-5420,MRMJ65602129091077
EMP2456198334,Lisa,Wang,lisa.wang@company.com,001-432-876-5432,Marketing,Marketing Specialist,2020-08-15,1700,58900.25,2.1,Active,1988-07-22,7890 Oak Street,New York,NY,10001,USA,F,Bachelor,85.5,2023-12-15,Junior,10,3,Office,Day,(212)555-1234,123-45-6789,ABCD12345678901234
EMP9876543210,Carlos,Rodriguez,carlos.r@company.com,555-123-4567,HR,HR Manager,2019-03-01,2200,72000.00,3.2,Active,1985-11-30,456 Pine Avenue,Chicago,IL,60007,USA,M,Master,90.2,2024-02-28,Senior,18,7,Hybrid,Day,(312)444-5678,987-65-4321,EFGH09876543210987"""
    
    with open(file_path, 'w') as f:
        f.write(data)
    
    return file_path


def test_load_data(sample_csv_file):
    """Test loading data from a CSV file."""
    parser = EmployeeDataParser()
    df = parser.load_data(sample_csv_file)
    
    # Verify that the dataframe was loaded correctly
    assert isinstance(df, pd.DataFrame)
    assert len(df) == 3  # There should be 3 records
    assert 'employee_id' in df.columns
    assert df['employee_id'].iloc[0] == 'EMP3479083960'


def test_load_data_nonexistent_file():
    """Test loading a non-existent file."""
    parser = EmployeeDataParser()
    with pytest.raises(FileNotFoundError):
        parser.load_data("archivo_inexistente.csv")


def test_load_data_no_file_path():
    """Test loading without providing a file path."""
    parser = EmployeeDataParser()
    with pytest.raises(ValueError):
        parser.load_data()


def test_clean_data(sample_csv_file):
    """Test cleaning data."""
    parser = EmployeeDataParser(sample_csv_file)
    parser.load_data()
    df = parser.clean_data()
    
    # Verify that the dates were converted correctly
    assert pd.api.types.is_datetime64_dtype(df['hire_date'])
    assert pd.api.types.is_datetime64_dtype(df['birth_date'])
    assert pd.api.types.is_datetime64_dtype(df['last_review_date'])
    
    # Verify that the numeric values were converted correctly
    assert pd.api.types.is_numeric_dtype(df['days_service'])
    assert pd.api.types.is_numeric_dtype(df['base_salary'])
    assert pd.api.types.is_numeric_dtype(df['bonus_percentage'])
    
    # Verify that the derived columns were created
    assert 'age' in df.columns
    assert 'total_compensation' in df.columns
    
    # Verify total compensation calculation
    expected_compensation = 65242.57 * (1 + 1.76/100)
    assert abs(df['total_compensation'].iloc[0] - expected_compensation) < 0.01


def test_clean_data_without_loading():
    """Test cleaning data without loading it previously."""
    parser = EmployeeDataParser()
    with pytest.raises(ValueError):
        parser.clean_data()


def test_get_data_without_loading():
    """Test getting data without loading it previously."""
    parser = EmployeeDataParser()
    with pytest.raises(ValueError):
        parser.get_data()
