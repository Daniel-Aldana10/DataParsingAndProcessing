import os
import pandas as pd
from pathlib import Path
from data_algorithms.data_parser import EmployeeDataParser
from data_algorithms.data_analyzer import EmployeeDataAnalyzer
from data_algorithms.data_visualizer import EmployeeDataVisualizer


def main():
    """
    Main function that executes the full analysis flow.
    """
    # Configure directories
    base_dir = Path(__file__).parent
    data_dir = base_dir / 'data'
    plots_dir = base_dir / 'plots'
    
    # Ensure directories exist
    plots_dir.mkdir(parents=True, exist_ok=True)
    
    # Fixed file name
    file = "data13_Small.csv"
    
    # Get data file path
    data_file = data_dir / file
    if not data_file.exists():
        print(f"Error: The file {data_file} does not exist.")
        print(f"Please place the CSV file in the {data_dir} directory.")
        return
    
    print("Starting employee data analysis...")
    
    # Step 1: Load and clean data
    print("\n1. Loading and cleaning data...")
    parser = EmployeeDataParser(data_file)
    df = parser.load_data()
    print(f"   - Data loaded. Initial dimensions: {df.shape}")

    clean_df = parser.clean_data()
    print(f"   - Data cleaned. Final dimensions: {clean_df.shape}")

    # Show a summary of the data
    print("\nPreview of the data:")
    print(clean_df.head())
    print("\nAvailable columns:")
    print(clean_df.columns.tolist())

    # Step 2: Perform analysis
    print("\n2. Performing analysis...")
    analyzer = EmployeeDataAnalyzer(clean_df)

    # General statistics
    print("\nGeneral statistics:")
    summary_stats = analyzer.get_summary_statistics()
    print(summary_stats)

    # Department analysis
    print("\nDepartment analysis:")
    dept_analysis = analyzer.analyze_by_department()
    for key, value in dept_analysis.items():
        print(f"\n{key}:")
        print(value)

    # Salary factors analysis
    print("\nSalary factors analysis:")
    salary_factors = analyzer.analyze_salary_factors()
    for key, value in salary_factors.items():
        print(f"\n{key}:")
        print(value)

    # Retention analysis
    print("\nRetention analysis:")
    retention_analysis = analyzer.analyze_retention()
    print(retention_analysis)

    # Salary equity analysis
    print("\nSalary equity analysis by gender:")
    equity_analysis = analyzer.analyze_salary_equity(group_col='gender')
    print(equity_analysis)

    # Education-Salary relationship analysis
    print("\nEducation-Salary relationship analysis:")
    education_salary_analysis = analyzer.analyze_education_salary_relationship()
    print("Education levels found:", list(education_salary_analysis['education_mapping'].keys()) if 'education_mapping' in education_salary_analysis else "N/A")
    if 'correlation' in education_salary_analysis:
        corr = education_salary_analysis['correlation']
        print(f"Correlation: {corr['correlation']:.4f} (p-value: {corr['p_value']:.4f})")
    if 'anova_test' in education_salary_analysis:
        anova = education_salary_analysis['anova_test']
        print(f"ANOVA F-statistic: {anova['f_statistic']:.4f} (p-value: {anova['p_value']:.4f})")
        print(f"Significant difference: {'Yes' if anova['significant'] else 'No'}")

    # Employee segmentation
    print("\nEmployee segmentation:")
    segments = analyzer.segment_employees(n_clusters=4)
    print(segments)

    # Step 3: Create visualizations
    print("\n3. Generating visualizations...")
    visualizer = EmployeeDataVisualizer(clean_df, plots_dir)
    
    # Create plots (without showing them, only saving them)
    print("   - Creating salary distribution plot...")
    visualizer.plot_salary_distribution(show=False)

    print("   - Creating department analysis plot...")
    visualizer.plot_department_analysis(show=False)

    print("   - Creating performance-salary scatter plot...")
    visualizer.plot_performance_salary_scatter(show=False)

    print("   - Creating service histogram...")
    visualizer.plot_service_histogram(show=False)

    print("   - Creating gender by department analysis plot...")
    visualizer.plot_gender_department_analysis(show=False)
    
    print("   - Creating age distribution by gender plot...")
    visualizer.plot_age_distribution_simple(show=False)
    
    print("   - Creating correlation matrix plot...")
    visualizer.plot_correlation_matrix(show=False)

    print("   - Creating salary vs experience by gender plot...")
    visualizer.plot_salary_experience_by_gender(show=False)

    print("   - Creating education by department plot...")
    visualizer.plot_education_by_department(show=False)
    
    print("   - Creating education-salary analysis plot...")
    visualizer.plot_education_salary_simple(show=False)

    print("\n¡Analysis completed successfully!")
    print(f"The visualizations have been saved in the directory: {plots_dir}")




if __name__ == "__main__":
    main()
