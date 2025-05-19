import pytest
import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from data_algorithms.data_visualizer import EmployeeDataVisualizer

@pytest.fixture
def sample_data():
    """Create sample data for testing visualizations"""
    np.random.seed(42)
    n_samples = 100
    
    data = {
        'employee_id': range(1, n_samples + 1),
        'base_salary': np.random.normal(50000, 10000, n_samples),
        'performance_score': np.random.uniform(1, 5, n_samples),
        'days_service': np.random.randint(1, 3650, n_samples),
        'department': np.random.choice(['HR', 'IT', 'Sales', 'Engineering'], n_samples),
        'gender': np.random.choice(['M', 'F'], n_samples),
        'age': np.random.normal(40, 10, n_samples),
        'education': np.random.choice(['Bachelor', 'Master', 'PhD'], n_samples)
    }
    
    return pd.DataFrame(data)

@pytest.fixture
def visualizer(sample_data, tmp_path):
    """Create a visualizer instance with sample data"""
    return EmployeeDataVisualizer(sample_data, output_dir=tmp_path)

def test_init_without_data():
    """Test initializing visualizer without data"""
    viz = EmployeeDataVisualizer()
    assert viz.data is None
    assert isinstance(viz.output_dir, Path)

def test_set_data(visualizer, sample_data):
    """Test setting data after initialization"""
    visualizer.set_data(sample_data)
    pd.testing.assert_frame_equal(visualizer.data, sample_data)

def test_set_data_invalid_type(visualizer):
    """Test setting invalid data type"""
    with pytest.raises(TypeError):
        visualizer.set_data([1, 2, 3])

def test_plot_salary_distribution(visualizer, tmp_path):
    """Test salary distribution plot creation"""
    fig = visualizer.plot_salary_distribution(save=True, show=False)
    assert isinstance(fig, plt.Figure)
    assert (tmp_path / 'salary_distribution.png').exists()
    plt.close(fig)

def test_plot_department_analysis(visualizer, tmp_path):
    """Test department analysis plot creation"""
    fig = visualizer.plot_department_analysis(save=True, show=False)
    assert isinstance(fig, plt.Figure)
    assert (tmp_path / 'department_analysis.png').exists()
    plt.close(fig)

def test_plot_performance_salary_scatter(visualizer, tmp_path):
    """Test performance vs salary scatter plot creation"""
    fig = visualizer.plot_performance_salary_scatter(save=True, show=False)
    assert isinstance(fig, plt.Figure)
    assert (tmp_path / 'performance_salary_scatter.png').exists()
    plt.close(fig)

def test_plot_service_histogram(visualizer, tmp_path):
    """Test service histogram plot creation"""
    fig = visualizer.plot_service_histogram(save=True, show=False)
    assert isinstance(fig, plt.Figure)
    assert (tmp_path / 'service_histogram.png').exists()
    plt.close(fig)

def test_plot_gender_department_analysis(visualizer, tmp_path):
    """Test gender department analysis plot creation"""
    fig = visualizer.plot_gender_department_analysis(save=True, show=False)
    assert isinstance(fig, plt.Figure)
    assert (tmp_path / 'gender_department_analysis.png').exists()
    plt.close(fig)

def test_plot_age_distribution_by_gender(visualizer, tmp_path):
    """Test age distribution by gender plot creation"""
    fig = visualizer.plot_age_distribution_by_gender(save=True, show=False)
    assert isinstance(fig, plt.Figure)
    assert (tmp_path / 'age_distribution_by_gender.png').exists()
    plt.close(fig)

def test_plot_correlation_matrix(visualizer, tmp_path):
    """Test correlation matrix plot creation"""
    fig = visualizer.plot_correlation_matrix(save=True, show=False)
    assert isinstance(fig, plt.Figure)
    assert (tmp_path / 'correlation_matrix.png').exists()
    plt.close(fig)

def test_plot_salary_experience_by_gender(visualizer, tmp_path):
    """Test salary vs experience by gender plot creation"""
    fig = visualizer.plot_salary_experience_by_gender(save=True, show=False)
    assert isinstance(fig, plt.Figure)
    assert (tmp_path / 'salary_experience_by_gender.png').exists()
    plt.close(fig)

def test_plot_education_by_department(visualizer, tmp_path):
    """Test education by department plot creation"""
    fig = visualizer.plot_education_by_department(save=True, show=False)
    assert isinstance(fig, plt.Figure)
    assert (tmp_path / 'education_by_department.png').exists()
    plt.close(fig)

def test_missing_required_columns(visualizer):
    """Test handling of missing required columns"""
    # Create data without required columns
    incomplete_data = pd.DataFrame({
        'employee_id': range(1, 101),
        'name': ['Employee ' + str(i) for i in range(1, 101)]
    })
    
    visualizer.set_data(incomplete_data)
    
    with pytest.raises(ValueError):
        visualizer.plot_salary_distribution()
    
    with pytest.raises(ValueError):
        visualizer.plot_gender_department_analysis()
    
    with pytest.raises(ValueError):
        visualizer.plot_age_distribution_by_gender()

def test_plot_with_no_data():
    """Test plotting without setting data"""
    visualizer = EmployeeDataVisualizer()
    
    with pytest.raises(ValueError):
        visualizer.plot_salary_distribution()
    
    with pytest.raises(ValueError):
        visualizer.plot_department_analysis()
    
    with pytest.raises(ValueError):
        visualizer.plot_performance_salary_scatter()

def test_custom_output_directory(sample_data, tmp_path):
    """Test using a custom output directory"""
    custom_dir = tmp_path / 'custom_plots'
    visualizer = EmployeeDataVisualizer(sample_data, output_dir=custom_dir)
    
    fig = visualizer.plot_salary_distribution(save=True, show=False)
    assert (custom_dir / 'salary_distribution.png').exists()
    plt.close(fig)

def test_plot_without_saving(visualizer):
    """Test plotting without saving to file"""
    fig = visualizer.plot_salary_distribution(save=False, show=False)
    assert isinstance(fig, plt.Figure)
    plt.close(fig) 