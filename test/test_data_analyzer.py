import pytest
import pandas as pd
import numpy as np
from data_algorithms.data_analyzer import EmployeeDataAnalyzer


@pytest.fixture
def sample_employee_data():
    """Creates a sample DataFrame for testing."""
    data = {
        'employee_id': ['EMP001', 'EMP002', 'EMP003', 'EMP004', 'EMP005', 'EMP006'],
        'first_name': ['John', 'Mary', 'Carlos', 'Lisa', 'Ahmed', 'Sarah'],
        'department': ['Engineering', 'Marketing', 'Engineering', 'HR', 'Marketing', 'HR'],
        'base_salary': [65000.0, 58000.0, 72000.0, 55000.0, 60000.0, 52000.0],
        'bonus_percentage': [2.0, 1.5, 3.0, 1.8, 2.2, 1.0],
        'performance_score': [75.0, 82.0, 90.0, 70.0, 85.0, 78.0],
        'days_service': [730, 365, 1095, 182, 547, 912],
        'gender': ['M', 'F', 'M', 'F', 'M', 'F'],
        'education': ['Bachelor', 'Master', 'PhD', 'Bachelor', 'Master', 'Bachelor'],
        'employee_level': ['Junior', 'Junior', 'Senior', 'Entry', 'Junior', 'Junior']
    }
    
    df = pd.DataFrame(data)
    
    # Create derived total compensation column
    df['total_compensation'] = df['base_salary'] * (1 + df['bonus_percentage'] / 100)
    
    return df


def test_set_data():
    """Test setting data in the analyzer."""
    analyzer = EmployeeDataAnalyzer()
    df = pd.DataFrame({'A': [1, 2, 3]})
    analyzer.set_data(df)
    assert analyzer.data is df


def test_set_data_wrong_type():
    """Test setting data with an incorrect type."""
    analyzer = EmployeeDataAnalyzer()
    with pytest.raises(TypeError):
        analyzer.set_data("not a dataframe")


def test_get_summary_statistics(sample_employee_data):
    """Test getting descriptive statistics."""
    analyzer = EmployeeDataAnalyzer(sample_employee_data)
    stats = analyzer.get_summary_statistics()
    
    # Verify that the statistics are correct
    assert isinstance(stats, pd.DataFrame)
    assert 'base_salary' in stats.columns
    assert 'performance_score' in stats.columns
    assert abs(stats['base_salary']['mean'] - sample_employee_data['base_salary'].mean()) < 0.001
    assert abs(stats['performance_score']['mean'] - sample_employee_data['performance_score'].mean()) < 0.001


def test_get_summary_statistics_no_data():
    """Test getting statistics without data."""
    analyzer = EmployeeDataAnalyzer()
    with pytest.raises(ValueError):
        analyzer.get_summary_statistics()


def test_analyze_by_department(sample_employee_data):
    """Test analyzing by department."""
    analyzer = EmployeeDataAnalyzer(sample_employee_data)
    dept_analysis = analyzer.analyze_by_department()
    
    # Verify that the analysis is correct
    assert isinstance(dept_analysis, dict)
    assert 'count' in dept_analysis
    assert 'avg_salary' in dept_analysis
    
    # Verify count by department
    assert dept_analysis['count']['Engineering'] == 2
    assert dept_analysis['count']['Marketing'] == 2
    assert dept_analysis['count']['HR'] == 2
    
    # Verify average salary by department
    eng_avg_salary = (65000.0 + 72000.0) / 2
    assert abs(dept_analysis['avg_salary']['Engineering'] - eng_avg_salary) < 0.001


def test_analyze_by_department_missing_column(sample_employee_data):
    """Test analyzing by department with missing column."""
    # Create a copy without the department column
    df_no_dept = sample_employee_data.drop('department', axis=1)
    analyzer = EmployeeDataAnalyzer(df_no_dept)
    
    with pytest.raises(ValueError):
        analyzer.analyze_by_department()



def test_analyze_salary_factors(sample_employee_data):
    """Test analyzing salary factors."""
    analyzer = EmployeeDataAnalyzer(sample_employee_data)
    salary_factors = analyzer.analyze_salary_factors()
    
    # Verify that the analysis is correct
    assert isinstance(salary_factors, dict)
    assert 'correlations' in salary_factors
    assert 'salary_by_gender' in salary_factors
    assert 'salary_by_education' in salary_factors
    
    # Verify that the correlations make sense
    assert 'performance_score' in salary_factors['correlations']
    assert 'days_service' in salary_factors['correlations']
    
    # Verify gender statistics
    assert 'mean' in salary_factors['salary_by_gender'].columns
    assert 'M' in salary_factors['salary_by_gender'].index
    assert 'F' in salary_factors['salary_by_gender'].index
    
    # Calculate manually to verify
    m_avg_salary = sample_employee_data[sample_employee_data['gender'] == 'M']['base_salary'].mean()
    assert abs(salary_factors['salary_by_gender'].loc['M', 'mean'] - m_avg_salary) < 0.001