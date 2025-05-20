import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path
from pandas.api.types import CategoricalDtype


class EmployeeDataVisualizer:
   
    
    def __init__(self, data=None, output_dir='./plots'):
        
        self.data = data
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Configure seaborn style for better visualizations
        sns.set_theme(style="whitegrid")
    
    def set_data(self, data):
       
        if not isinstance(data, pd.DataFrame):
            raise TypeError("Data must be a pandas DataFrame")
        self.data = data
    
    def plot_salary_distribution(self, save=True, show=True):
        
        if self.data is None:
            raise ValueError("No data has been set. Use set_data() first.")
        
        if 'base_salary' not in self.data.columns:
            raise ValueError("Data does not contain 'base_salary' column")
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Create histogram with KDE curve
        sns.histplot(data=self.data, x='base_salary', kde=True, ax=ax)
        
        # Add vertical lines for mean and median
        mean_salary = self.data['base_salary'].mean()
        median_salary = self.data['base_salary'].median()
        
        ax.axvline(mean_salary, color='red', linestyle='--', label=f'Mean: ${mean_salary:.2f}')
        ax.axvline(median_salary, color='green', linestyle='--', label=f'Median: ${median_salary:.2f}')
        
        # Configure plot
        ax.set_title('Base Salary Distribution', fontsize=15)
        ax.set_xlabel('Base Salary ($)', fontsize=12)
        ax.set_ylabel('Frequency', fontsize=12)
        ax.legend()
        
        # Save plot if requested
        if save:
            fig.savefig(self.output_dir / 'salary_distribution.png', dpi=300, bbox_inches='tight')
        
        if not show:
            plt.close(fig)
        
        return fig
    
    def plot_department_analysis(self, save=True, show=True):
      
        if self.data is None:
            raise ValueError("No data has been set. Use set_data() first.")
        
        if 'department' not in self.data.columns or 'base_salary' not in self.data.columns:
            missing = []
            if 'department' not in self.data.columns:
                missing.append('department')
            if 'base_salary' not in self.data.columns:
                missing.append('base_salary')
            raise ValueError(f"Data does not contain columns: {', '.join(missing)}")
        
        # Calculate department statistics
        dept_stats = self.data.groupby('department').agg({
            'base_salary': ['mean', 'count'],
            'performance_score': 'mean'
        })
        dept_stats.columns = ['salary_mean', 'employee_count', 'performance_mean']
        dept_stats = dept_stats.reset_index()
        
        # Sort by average salary descending
        dept_stats = dept_stats.sort_values('salary_mean', ascending=False)
        
        fig, ax1 = plt.subplots(figsize=(12, 8))
        
        # Create bars for average salary
        bars = ax1.bar(dept_stats['department'], dept_stats['salary_mean'], 
                       color='skyblue', alpha=0.7)
        
        # Configure primary Y axis
        ax1.set_xlabel('Department', fontsize=12)
        ax1.set_ylabel('Average Salary ($)', fontsize=12)
        ax1.tick_params(axis='x', rotation=45)
        
        # Create secondary Y axis for average performance
        ax2 = ax1.twinx()
        line = ax2.plot(dept_stats['department'], dept_stats['performance_mean'], 
                        marker='o', color='green', label='Average Performance')
        ax2.set_ylabel('Performance Score', fontsize=12)
        
        # Add employee count as text on bars
        for i, (bar, count) in enumerate(zip(bars, dept_stats['employee_count'])):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height + 1000,
                     f'n={count}', ha='center', va='bottom', fontsize=8)
        
        # Configure title and legend
        plt.title('Average Salary and Performance by Department', fontsize=15)
        
        # Create combined legend
        from matplotlib.lines import Line2D
        legend_elements = [
            Line2D([0], [0], color='skyblue', lw=4, label='Average Salary'),
            Line2D([0], [0], marker='o', color='green', label='Average Performance')
        ]
        ax1.legend(handles=legend_elements, loc='upper right')
        
        plt.tight_layout()
        
        # Save plot if requested
        if save:
            fig.savefig(self.output_dir / 'department_analysis.png', dpi=300, bbox_inches='tight')
        
        if not show:
            plt.close(fig)
        
        return fig
    
    def plot_performance_salary_scatter(self, save=True, show=True):
      
        if self.data is None:
            raise ValueError("No data has been set. Use set_data() first.")
        
        if 'performance_score' not in self.data.columns or 'base_salary' not in self.data.columns:
            missing = []
            if 'performance_score' not in self.data.columns:
                missing.append('performance_score')
            if 'base_salary' not in self.data.columns:
                missing.append('base_salary')
            raise ValueError(f"Data does not contain columns: {', '.join(missing)}")
        
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # Create scatter plot with regression line
        if 'department' in self.data.columns:
            scatter = sns.scatterplot(
                data=self.data,
                x='performance_score',
                y='base_salary',
                hue='department',
                size='days_service' if 'days_service' in self.data.columns else None,
                sizes=(20, 200) if 'days_service' in self.data.columns else None,
                alpha=0.7,
                ax=ax
            )
            
            # Adjust legend to not be too large
            plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        else:
            scatter = sns.scatterplot(
                data=self.data,
                x='performance_score',
                y='base_salary',
                size='days_service' if 'days_service' in self.data.columns else None,
                sizes=(20, 200) if 'days_service' in self.data.columns else None,
                alpha=0.7,
                ax=ax
            )
        
        # Add regression line
        sns.regplot(
            data=self.data,
            x='performance_score',
            y='base_salary',
            scatter=False,
            ax=ax,
            line_kws={'color': 'red'}
        )
        
        # Calculate and show the correlation
        corr = self.data['performance_score'].corr(self.data['base_salary'])
        ax.text(0.05, 0.95, f'Correlation: {corr:.2f}', 
                transform=ax.transAxes, fontsize=12,
                bbox=dict(facecolor='white', alpha=0.5))
        
        # Configure the plot
        ax.set_title('Relationship between Performance and Salary', fontsize=15)
        ax.set_xlabel('Performance Score', fontsize=12)
        ax.set_ylabel('Base Salary ($)', fontsize=12)
        
        plt.tight_layout()
        
        # Save the plot if requested
        if save:
            fig.savefig(self.output_dir / 'performance_salary_scatter.png', dpi=300, bbox_inches='tight')
        
        if not show:
            plt.close(fig)
        
        return fig
    
    def plot_service_histogram(self, save=True, show=True):
      
        if self.data is None:
            raise ValueError("No data has been set. Use set_data() first.")
        
        if 'days_service' not in self.data.columns:
            raise ValueError("Data does not contain 'days_service' column")
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Convert days to years for better interpretation
        self.data['years_service'] = self.data['days_service'] / 365.25
        
        # Create histogram with bins of 1 year
        bins = np.arange(0, self.data['years_service'].max() + 1, 1)
        sns.histplot(data=self.data, x='years_service', bins=bins, kde=True, ax=ax)
        
        # Configure the plot
        ax.set_title('Distribution of Employee Tenure', fontsize=15)
        ax.set_xlabel('Years of Service', fontsize=12)
        ax.set_ylabel('Number of Employees', fontsize=12)
        
        # Save the plot if requested
        if save:
            fig.savefig(self.output_dir / 'service_histogram.png', dpi=300, bbox_inches='tight')
        
        if not show:
            plt.close(fig)
        
        return fig

    def plot_gender_department_analysis(self, save=True, show=True):
      
        if self.data is None:
            raise ValueError("No data has been set. Use set_data() first.")
        
        if 'gender' not in self.data.columns or 'department' not in self.data.columns:
            missing = []
            if 'gender' not in self.data.columns:
                missing.append('gender')
            if 'department' not in self.data.columns:
                missing.append('department')
            raise ValueError(f"Data does not contain columns: {', '.join(missing)}")
        
        # Calculate gender percentages by department
        gender_dept = pd.crosstab(self.data['department'], self.data['gender'], normalize='index') * 100
        
        # Create plot
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 12))
        
        # Stacked bar chart (percentages)
        gender_dept.plot(kind='bar', stacked=True, ax=ax1)
        ax1.set_title('Gender Distribution by Department (%)', fontsize=15)
        ax1.set_xlabel('Department', fontsize=12)
        ax1.set_ylabel('Percentage', fontsize=12)
        ax1.tick_params(axis='x', rotation=45)
        
        # Add absolute numbers
        gender_abs = pd.crosstab(self.data['department'], self.data['gender'])
        gender_abs.plot(kind='bar', ax=ax2)
        ax2.set_title('Number of Employees by Gender and Department', fontsize=15)
        ax2.set_xlabel('Department', fontsize=12)
        ax2.set_ylabel('Number of Employees', fontsize=12)
        ax2.tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        
        if save:
            fig.savefig(self.output_dir / 'gender_department_analysis.png', dpi=300, bbox_inches='tight')
        
        if not show:
            plt.close(fig)
        
        return fig

    def plot_age_distribution_by_gender(self, save=True, show=True):
        if self.data is None:
            raise ValueError("No data has been set. Use set_data() first.")
        
        if 'age' not in self.data.columns or 'gender' not in self.data.columns:
            missing = []
            if 'age' not in self.data.columns:
                missing.append('age')
            if 'gender' not in self.data.columns:
                missing.append('gender')
            raise ValueError(f"Data does not contain columns: {', '.join(missing)}")
        
        fig, ax = plt.subplots(figsize=(10, 6))

        # Create violin plot
        sns.violinplot(data=self.data, x='gender', y='age', ax=ax)
        
        # Add swarm plot to see the distribution of points
        sns.swarmplot(data=self.data, x='gender', y='age', color='white', alpha=0.5, size=3, ax=ax)
        
        # Configure the plot
        ax.set_title('Age Distribution by Gender', fontsize=15)
        ax.set_xlabel('Gender', fontsize=12)
        ax.set_ylabel('Age', fontsize=12)
        
        # Add basic statistics
        for i, gender in enumerate(self.data['gender'].unique()):
            stats = self.data[self.data['gender'] == gender]['age'].agg(['mean', 'median'])
            ax.text(i, self.data['age'].max(), f'Mean: {stats["mean"]:.1f}\nMedian: {stats["median"]:.1f}',
                    ha='center', va='bottom')
        
        if save:
            fig.savefig(self.output_dir / 'age_distribution_by_gender.png', dpi=300, bbox_inches='tight')
        
        if not show:
            plt.close(fig)
        
        return fig

    def plot_age_distribution_simple(self, save=True, show=True):

        if self.data is None:
            raise ValueError("No data has been set. Use set_data() first.")
        
        if 'age' not in self.data.columns or 'gender' not in self.data.columns:
            missing = []
            if 'age' not in self.data.columns:
                missing.append('age')
            if 'gender' not in self.data.columns:
                missing.append('gender')
            raise ValueError(f"Data does not contain columns: {', '.join(missing)}")
        
        fig, ax = plt.subplots(figsize=(10, 6))

        # Create boxplot instead of violin+swarm (much faster)
        sns.boxplot(data=self.data, x='gender', y='age', ax=ax)
        
        # Configure the plot
        ax.set_title('Age Distribution by Gender', fontsize=15)
        ax.set_xlabel('Gender', fontsize=12)
        ax.set_ylabel('Age', fontsize=12)
        
        # Add basic statistics
        stats_by_gender = self.data.groupby('gender')['age'].agg(['mean', 'median', 'count'])
        
        # Add text with statistics
        y_pos = self.data['age'].max() * 0.95
        for i, gender in enumerate(stats_by_gender.index):
            stats = stats_by_gender.loc[gender]
            ax.text(i, y_pos, 
                   f"Mean: {stats['mean']:.1f}\nMedian: {stats['median']:.1f}\nCount: {stats['count']}",
                   ha='center', va='top', 
                   bbox=dict(facecolor='white', alpha=0.8))
        

        if save:
            fig.savefig(self.output_dir / 'age_distribution_by_gender.png', dpi=150, bbox_inches='tight')
        
        if not show:
            plt.close(fig)
        
        return fig

    def plot_correlation_matrix(self, save=True, show=True):
       
        if self.data is None:
            raise ValueError("No data has been set. Use set_data() first.")
        
        # Select only numeric columns
        numeric_data = self.data.select_dtypes(include=[np.number])
        
        # Calculate correlation matrix
        corr_matrix = numeric_data.corr()
        
        # Create plot
        fig, ax = plt.subplots(figsize=(12, 10))
        
        # Create heatmap
        sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0, ax=ax)
        
        # Configure the plot
        ax.set_title('Correlation Matrix of Numeric Variables', fontsize=15)
        
        plt.tight_layout()
        
        if save:
            fig.savefig(self.output_dir / 'correlation_matrix.png', dpi=300, bbox_inches='tight')
        
        if not show:
            plt.close(fig)
        
        return fig

    def plot_salary_experience_by_gender(self, save=True, show=True):
        
        if self.data is None:
            raise ValueError("No data has been set. Use set_data() first.")
        
        required_cols = ['base_salary', 'days_service', 'gender']
        missing = [col for col in required_cols if col not in self.data.columns]
        if missing:
            raise ValueError(f"Data does not contain columns: {', '.join(missing)}")
        
        # Convert days to years
        self.data['years_service'] = self.data['days_service'] / 365.25
        
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # Create scatter plot by gender
        for gender in self.data['gender'].unique():
            gender_data = self.data[self.data['gender'] == gender]
            ax.scatter(gender_data['years_service'], gender_data['base_salary'],
                      alpha=0.6, label=gender)
            
            # Add trend line
            z = np.polyfit(gender_data['years_service'], gender_data['base_salary'], 1)
            p = np.poly1d(z)
            ax.plot(gender_data['years_service'], p(gender_data['years_service']),
                   linestyle='--', alpha=0.8)
        
        # Configure the plot
        ax.set_title('Salary vs Years of Experience by Gender', fontsize=15)
        ax.set_xlabel('Years of Service', fontsize=12)
        ax.set_ylabel('Base Salary ($)', fontsize=12)
        ax.legend()
        
        if save:
            fig.savefig(self.output_dir / 'salary_experience_by_gender.png', dpi=300, bbox_inches='tight')
        
        if not show:
            plt.close(fig)
        
        return fig

    def plot_education_by_department(self, save=True, show=True):
        
        if self.data is None:
            raise ValueError("No data has been set. Use set_data() first.")
        
        if 'education' not in self.data.columns or 'department' not in self.data.columns:
            missing = []
            if 'education' not in self.data.columns:
                missing.append('education')
            if 'department' not in self.data.columns:
                missing.append('department')
            raise ValueError(f"Data does not contain columns: {', '.join(missing)}")
        
        # Calculate education percentages by department
        edu_dept = pd.crosstab(self.data['department'], self.data['education'], normalize='index') * 100
        
        fig, ax = plt.subplots(figsize=(12, 6))
        
        # Create stacked bar chart
        edu_dept.plot(kind='bar', stacked=True, ax=ax)
        
        # Configure the plot
        ax.set_title('Education Level Distribution by Department', fontsize=15)
        ax.set_xlabel('Department', fontsize=12)
        ax.set_ylabel('Percentage', fontsize=12)
        ax.tick_params(axis='x', rotation=45)
        ax.legend(title='Education Level', bbox_to_anchor=(1.05, 1), loc='upper left')
        
        plt.tight_layout()
        
        if save:
            fig.savefig(self.output_dir / 'education_by_department.png', dpi=300, bbox_inches='tight')
        
        if not show:
            plt.close(fig)
        
        return fig
        
    def plot_education_salary_simple(self, save=True, show=True):

        if self.data is None:
            raise ValueError("No data has been set. Use set_data() first.")
        
        if 'education' not in self.data.columns or 'base_salary' not in self.data.columns:
            missing = []
            if 'education' not in self.data.columns:
                missing.append('education')
            if 'base_salary' not in self.data.columns:
                missing.append('base_salary')
            raise ValueError(f"Data does not contain columns: {', '.join(missing)}")
        
        # Create figure with two subplots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        
        # 1. Boxplot of salary by education level (without swarmplot)
        try:
            education_order = ['High School', 'Associate', 'Bachelor', 'Master', 'PhD']
            edu_levels = self.data['education'].unique()
            education_order = [level for level in education_order if level in edu_levels]
            if len(education_order) != len(edu_levels):
                education_order = sorted(edu_levels)
        except:
            education_order = self.data['education'].unique()
        
        # Simple boxplot without individual points
        sns.boxplot(data=self.data, x='education', y='base_salary', order=education_order, ax=ax1)
        
        # Configure the boxplot
        ax1.set_title('Salary Distribution by Education Level', fontsize=15)
        ax1.set_xlabel('Education Level', fontsize=12)
        ax1.set_ylabel('Base Salary ($)', fontsize=12)
        ax1.tick_params(axis='x', rotation=45)
        
        # 2. Bar plot of average salary by education level
        # Use a simpler aggregation approach
        edu_salary = self.data.groupby('education')['base_salary'].agg(['mean', 'count'])
        edu_salary = edu_salary.sort_values('mean', ascending=False)
        
        # Create bar plot
        ax2.bar(edu_salary.index, edu_salary['mean'], color='skyblue')
        
        # Configure the bar plot
        ax2.set_title('Average Salary by Education Level', fontsize=15)
        ax2.set_xlabel('Education Level', fontsize=12)
        ax2.set_ylabel('Average Salary ($)', fontsize=12)
        ax2.tick_params(axis='x', rotation=45)
        
        # Add correlation if we can calculate it
        try:
            # Create a simple education level mapping (1 to n)
            edu_mapping = {level: i+1 for i, level in enumerate(sorted(self.data['education'].unique()))}
            temp_df = self.data.copy()
            temp_df['edu_numeric'] = temp_df['education'].map(edu_mapping)
            
            # Calculate correlation
            corr = temp_df['edu_numeric'].corr(temp_df['base_salary'])
            corr_text = f'Correlation: {corr:.2f}'
            
            # Add text to the second subplot
            ax2.text(0.05, 0.95, corr_text, transform=ax2.transAxes, 
                    fontsize=12, bbox=dict(facecolor='white', alpha=0.5))
        except:
            pass
        
        plt.tight_layout()
        
        # Usar DPI más bajo para acelerar el guardado
        if save:
            fig.savefig(self.output_dir / 'education_salary_analysis.png', dpi=150, bbox_inches='tight')
        
        if not show:
            plt.close(fig)
        
        return fig