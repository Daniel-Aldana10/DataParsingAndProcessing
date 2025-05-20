import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
import scipy.stats as stats
from datetime import datetime


class EmployeeDataAnalyzer:
  
    def __init__(self, data=None):
      
        self.data = data
    
    def set_data(self, data):
        if not isinstance(data, pd.DataFrame):
            raise TypeError("Data must be a pandas DataFrame")
        self.data = data
    
    def get_summary_statistics(self):
        if self.data is None:
            raise ValueError("No data has been set. Use set_data() first.")
        
        # Select numeric columns
        numeric_data = self.data.select_dtypes(include=['number'])
        
        # Calculate descriptive statistics
        summary = numeric_data.describe()
        
        return summary
    
    def analyze_by_department(self):
        if self.data is None:
            raise ValueError("No data has been set. Use set_data() first.")
        
        if 'department' not in self.data.columns:
            raise ValueError("Data does not contain 'department' column")
        
        result = {
            'count': self.data.groupby('department').size(),
            'avg_salary': self.data.groupby('department')['base_salary'].mean(),
            'avg_performance': self.data.groupby('department')['performance_score'].mean(),
            'avg_service_days': self.data.groupby('department')['days_service'].mean()
        }
        
        # Calculate total_compensation if exists
        if 'total_compensation' in self.data.columns:
            result['avg_total_compensation'] = self.data.groupby('department')['total_compensation'].mean()
        
        return result
    
    def analyze_by_job_level(self):
        if self.data is None:
            raise ValueError("No data has been set. Use set_data() first.")
        
        if 'employee_level' not in self.data.columns:
            raise ValueError("Data does not contain 'employee_level' column")
        
        result = {
            'count': self.data.groupby('employee_level').size(),
            'avg_salary': self.data.groupby('employee_level')['base_salary'].mean(),
            'avg_performance': self.data.groupby('employee_level')['performance_score'].mean()
        }
        
        return result
    
    def analyze_salary_factors(self):
        if self.data is None:
            raise ValueError("No data has been set. Use set_data() first.")
        
        if 'base_salary' not in self.data.columns:
            raise ValueError("Data does not contain 'base_salary' column")
        
        # Select relevant numeric columns for correlation
        numeric_cols = ['days_service', 'performance_score', 'age', 'base_salary']
        available_cols = [col for col in numeric_cols if col in self.data.columns]
        
        result = {
            'correlations': self.data[available_cols].corr()['base_salary'].drop('base_salary')
        }
        
        # Analysis by gender if available
        if 'gender' in self.data.columns and 'base_salary' in self.data.columns:
            result['salary_by_gender'] = self.data.groupby('gender')['base_salary'].agg(['mean', 'median', 'std'])
        
        # Analysis by education if available
        if 'education' in self.data.columns and 'base_salary' in self.data.columns:
            result['salary_by_education'] = self.data.groupby('education')['base_salary'].agg(['mean', 'median', 'std'])
        
        return result
    
    def analyze_temporal_trends(self, date_col='hire_date', value_col='base_salary', freq='Y'):
        if self.data is None:
            raise ValueError("No data has been set. Use set_data() first.")
        
        if date_col not in self.data.columns:
            raise ValueError(f"Data does not contain column '{date_col}'")
        if value_col not in self.data.columns:
            raise ValueError(f"Data does not contain column '{value_col}'")
        
        # Ensure date column is datetime type
        if not pd.api.types.is_datetime64_dtype(self.data[date_col]):
            try:
                date_series = pd.to_datetime(self.data[date_col])
            except:
                raise ValueError(f"Column '{date_col}' cannot be converted to date.")
        else:
            date_series = self.data[date_col]
        
        # Create temporary DataFrame for analysis
        temp_df = pd.DataFrame({
            'date': date_series,
            'value': self.data[value_col]
        }).dropna()
        
        # Set date as index for resampling
        temp_df.set_index('date', inplace=True)
        
        # Calculate statistics by time period
        trends = temp_df.resample(freq).agg(['mean', 'median', 'std', 'count'])
        
        # Calculate growth vs previous period
        trends['growth'] = trends['value']['mean'].pct_change() * 100
        
        # Reorganize columns for better readability
        trends = trends['value']
        trends.columns = ['mean', 'median', 'std', 'count', 'growth_pct']
        
        return trends
    
    def predict_salary(self, features=None):
        if self.data is None:
            raise ValueError("No data has been set. Use set_data() first.")
        
        if 'base_salary' not in self.data.columns:
            raise ValueError("Data does not contain 'base_salary' column")
        
        # If no features specified, use all numeric columns except salary
        if features is None:
            features = [col for col in self.data.select_dtypes(include=['number']).columns 
                       if col != 'base_salary']
        
        # Prepare data
        X = self.data[features].copy()
        y = self.data['base_salary']
        
        # Handle missing values
        X = X.fillna(X.mean())
        
        # Create and fit model
        model = LinearRegression()
        model.fit(X, y)
        
        # Get predictions
        y_pred = model.predict(X)
        
        # Calculate metrics
        r2 = model.score(X, y)
        mse = np.mean((y - y_pred) ** 2)
        rmse = np.sqrt(mse)
        
        # Create feature importance DataFrame
        feature_importance = pd.DataFrame({
            'feature': features,
            'coefficient': model.coef_
        }).sort_values('coefficient', key=abs, ascending=False)
        
        result = {
            'r2_score': r2,
            'rmse': rmse,
            'feature_importance': feature_importance,
            'intercept': model.intercept_,
            'predictions': y_pred
        }
        
        return result
    
    def segment_employees(self, n_clusters=3, features=None):
        if self.data is None:
            raise ValueError("No se han establecido datos. Usa set_data() primero.")
        
        # If no features specified, use some default ones
        if features is None:
            potential_features = ['base_salary', 'performance_score', 'days_service', 
                               'age', 'bonus_percentage', 'total_compensation']
            features = [f for f in potential_features if f in self.data.columns]
        
        # Verify that there are enough features for segmentation
        if not features or len(features) < 2:
            raise ValueError("Not enough features for segmentation")
        
        # Create DataFrame for segmentation
        X = self.data[features].copy()
        
        # Handle missing values
        X = X.fillna(X.mean())
        
        # Standardize variables to have the same scale
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Apply K-means
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        clusters = kmeans.fit_predict(X_scaled)
        
        # Add cluster column to original DataFrame
        result_df = self.data.copy()
        result_df['cluster'] = clusters
        
        # Calculate features of each cluster
        cluster_profiles = result_df.groupby('cluster')[features].mean()
        
        # Calculate size of each cluster
        cluster_sizes = result_df['cluster'].value_counts().sort_index()
        
        return {
            'segmented_data': result_df,
            'cluster_profiles': cluster_profiles,
            'cluster_sizes': cluster_sizes,
            'model': kmeans,
            'features_used': features
        }
    
    def analyze_retention(self, target_col=None):
        if self.data is None:
            raise ValueError("No data has been set. Use set_data() first.")
        
        # If no target column specified, try to find an appropriate one
        if target_col is None:
            potential_cols = ['status', 'employee_status', 'active']
            for col in potential_cols:
                if col in self.data.columns:
                    target_col = col
                    break
        
        if target_col is None or target_col not in self.data.columns:
            raise ValueError("No valid column found for retention analysis")
        
        # Create a binary retention column if it's not already
        retention_df = self.data.copy()
        
        # Determine active vs inactive based on common values
        active_values = ['Active', 'active', 'ACTIVE', 'A', 1, True, 'Yes', 'Y', 'yes', 'y', 'Activo']
        inactive_values = ['Inactive', 'inactive', 'INACTIVE', 'I', 0, False, 'No', 'N', 'no', 'n', 'Inactivo', 'Terminated', 'terminated', 'TERMINATED']
        
        if pd.api.types.is_object_dtype(retention_df[target_col]):
            retention_df['is_active'] = retention_df[target_col].apply(
                lambda x: 1 if str(x) in [str(v) for v in active_values] else
                         (0 if str(x) in [str(v) for v in inactive_values] else None)
            )
        else:
            # If already numeric, convert to binary
            retention_df['is_active'] = retention_df[target_col].apply(
                lambda x: 1 if x in active_values else
                         (0 if x in inactive_values else None)
            )
        
        # Eliminate rows where status couldn't be determined
        retention_df = retention_df.dropna(subset=['is_active'])
        
        # Calculate global retention rate
        total_employees = len(retention_df)
        active_employees = retention_df['is_active'].sum()
        retention_rate = (active_employees / total_employees) * 100
        
        # Calculate retention rates by department if available
        dept_retention = None
        if 'department' in retention_df.columns:
            dept_retention = retention_df.groupby('department')['is_active'].agg(['mean', 'count'])
            dept_retention['retention_rate'] = dept_retention['mean'] * 100
            dept_retention = dept_retention.sort_values('retention_rate', ascending=False)
        
        # Calculate retention rates by employee level if available
        level_retention = None
        if 'employee_level' in retention_df.columns:
            level_retention = retention_df.groupby('employee_level')['is_active'].agg(['mean', 'count'])
            level_retention['retention_rate'] = level_retention['mean'] * 100
            level_retention = level_retention.sort_values('retention_rate', ascending=False)
        
        # Calculate correlation between retention and other numeric variables
        numeric_cols = retention_df.select_dtypes(include=['number']).columns.tolist()
        numeric_cols = [col for col in numeric_cols if col != 'is_active']
        retention_correlations = {}
        
        for col in numeric_cols:
            if col in retention_df.columns:
                correlation = retention_df[['is_active', col]].corr().iloc[0, 1]
                if not pd.isna(correlation):
                    retention_correlations[col] = correlation
        
        # Sort correlations by absolute value
        retention_correlations = {k: v for k, v in sorted(
            retention_correlations.items(), 
            key=lambda item: abs(item[1]), 
            reverse=True
        )}
        
        # Calculate average service time for active vs inactive employees
        service_comparison = None
        if 'days_service' in retention_df.columns:
            service_comparison = retention_df.groupby('is_active')['days_service'].agg(['mean', 'median', 'std'])
            
        # Calculate basic survival curves
        survival_data = None
        if 'hire_date' in retention_df.columns:
            # Calculate tenure for each employee
            current_date = datetime.now().date()
            retention_df['tenure_days'] = (pd.to_datetime(current_date) - retention_df['hire_date']).dt.days
            
            # Group by service years
            retention_df['tenure_years'] = retention_df['tenure_days'] / 365.25
            retention_df['tenure_year_bucket'] = pd.cut(
                retention_df['tenure_years'], 
                bins=[0, 1, 2, 3, 5, 10, 100],
                labels=['0-1', '1-2', '2-3', '3-5', '5-10', '10+']
            )
            
            survival_data = retention_df.groupby('tenure_year_bucket')['is_active'].agg(['mean', 'count'])
            survival_data['retention_rate'] = survival_data['mean'] * 100
        
        return {
            'overall_retention_rate': retention_rate,
            'active_count': active_employees,
            'inactive_count': total_employees - active_employees,
            'total_count': total_employees,
            'department_retention': dept_retention,
            'level_retention': level_retention,
            'retention_correlations': retention_correlations,
            'service_comparison': service_comparison,
            'survival_curve': survival_data
        }
    
    def analyze_salary_equity(self, group_col='gender'):
        if self.data is None:
            raise ValueError("No data has been set. Use set_data() first.")
        
        if 'base_salary' not in self.data.columns:
            raise ValueError("Data does not contain 'base_salary' column")
        
        if group_col not in self.data.columns:
            raise ValueError(f"Data does not contain column '{group_col}'")
        
        # Filter complete data for analysis
        analysis_df = self.data.dropna(subset=['base_salary', group_col])
        
        # Calculate basic statistics by group
        group_stats = analysis_df.groupby(group_col)['base_salary'].agg(['mean', 'median', 'std', 'count'])
        
        # Sort groups by average salary in descending order
        group_stats = group_stats.sort_values('mean', ascending=False)
        
        # Calculate salary gap between groups
        if len(group_stats) > 1:
            highest_group = group_stats.index[0]
            highest_mean = group_stats.loc[highest_group, 'mean']
            
            # Calculate percentage gap relative to the highest salary group
            group_stats['gap_pct'] = ((highest_mean - group_stats['mean']) / highest_mean) * 100
        
        # Perform department analysis if available
        dept_analysis = None
        if 'department' in analysis_df.columns:
            dept_analysis = {}
            departments = analysis_df['department'].unique()
            
            for dept in departments:
                dept_data = analysis_df[analysis_df['department'] == dept]
                if len(dept_data) > 10:  # Analyze only departments with enough data
                    dept_stats = dept_data.groupby(group_col)['base_salary'].agg(['mean', 'median', 'count'])
                    if len(dept_stats) > 1:  # Only if there are at least two groups to compare
                        dept_analysis[dept] = dept_stats
        
        # Perform employee level analysis if available
        level_analysis = None
        if 'employee_level' in analysis_df.columns:
            level_analysis = {}
            levels = analysis_df['employee_level'].unique()
            
            for level in levels:
                level_data = analysis_df[analysis_df['employee_level'] == level]
                if len(level_data) > 10:  # Analyze only levels with enough data
                    level_stats = level_data.groupby(group_col)['base_salary'].agg(['mean', 'median', 'count'])
                    if len(level_stats) > 1:  # Only if there are at least two groups to compare
                        level_analysis[level] = level_stats
        
        # Perform statistical test to see if differences are significant
        t_test_results = None
        p_values = []
        
        # Only perform test if there are at least two groups
        if len(analysis_df[group_col].unique()) > 1:
            groups = analysis_df[group_col].unique()
            t_test_results = {}
            
            # Compare each group with the others
            for i, group1 in enumerate(groups):
                group1_data = analysis_df[analysis_df[group_col] == group1]['base_salary'].dropna()
                
                for group2 in groups[i+1:]:
                    group2_data = analysis_df[analysis_df[group_col] == group2]['base_salary'].dropna()
                    
                    # Only perform test if there are enough data
                    if len(group1_data) > 5 and len(group2_data) > 5:
                        t_stat, p_val = stats.ttest_ind(group1_data, group2_data, equal_var=False)
                        
                        t_test_results[f"{group1} vs {group2}"] = {
                            't_statistic': t_stat,
                            'p_value': p_val,
                            'significant': p_val < 0.05
                        }
                        
                        p_values.append(p_val)
        
        return {
            'group_stats': group_stats,
            'department_analysis': dept_analysis,
            'level_analysis': level_analysis,
            'statistical_tests': t_test_results,
            'has_significant_gap': any(p < 0.05 for p in p_values) if p_values else None
        }
    
    def run_statistical_tests(self, var1, var2, test_type='correlation'):
        if self.data is None:
            raise ValueError("No data has been set. Use set_data() first.")
        
        if var1 not in self.data.columns or var2 not in self.data.columns:
            missing = []
            if var1 not in self.data.columns:
                missing.append(var1)
            if var2 not in self.data.columns:
                missing.append(var2)
            raise ValueError(f"Variables {', '.join(missing)} do not exist in data")
        
        # Filter data without null values for analysis
        valid_data = self.data.dropna(subset=[var1, var2])
        
        if len(valid_data) == 0:
            raise ValueError("No valid data for analysis after removing nulls")
        
        results = {'test_type': test_type}
        
        if test_type == 'correlation':
            # For correlation, both variables must be numeric
            if not pd.api.types.is_numeric_dtype(valid_data[var1]) or not pd.api.types.is_numeric_dtype(valid_data[var2]):
                raise ValueError("Both variables must be numeric for correlation")
            
            # Calculate Pearson correlation
            corr, p_value = stats.pearsonr(valid_data[var1], valid_data[var2])
            
            results.update({
                'correlation': corr,
                'p_value': p_value,
                'significant': p_value < 0.05,
                'interpretation': self._interpret_correlation(corr, p_value)
            })
            
        elif test_type == 'ttest':
            # For t-test, var1 must be categorical (with 2 levels) and var2 numeric
            if not pd.api.types.is_numeric_dtype(valid_data[var2]):
                raise ValueError("The dependent variable must be numeric for t-test")
            
            # Get unique groups
            groups = valid_data[var1].unique()
            
            if len(groups) != 2:
                raise ValueError("The grouping variable must have exactly 2 levels for t-test")
            
            # Separate data by group
            group1 = valid_data[valid_data[var1] == groups[0]][var2]
            group2 = valid_data[valid_data[var1] == groups[1]][var2]
            
            # Perform t-test
            t_stat, p_value = stats.ttest_ind(group1, group2, equal_var=False)
            
            # Calculate group means
            mean1 = group1.mean()
            mean2 = group2.mean()
            
            results.update({
                'groups': {str(groups[0]): len(group1), str(groups[1]): len(group2)},
                'means': {str(groups[0]): mean1, str(groups[1]): mean2},
                'difference': mean1 - mean2,
                't_statistic': t_stat,
                'p_value': p_value,
                'significant': p_value < 0.05,
                'interpretation': self._interpret_ttest(p_value, mean1, mean2, groups[0], groups[1])
            })
            
        elif test_type == 'chisquare':
            # For chi-square, both variables must be categorical
            # Create contingency table
            contingency_table = pd.crosstab(valid_data[var1], valid_data[var2])
            
            # Perform chi-square test
            chi2, p_value, dof, expected = stats.chi2_contingency(contingency_table)
            
            results.update({
                'contingency_table': contingency_table,
                'chi2': chi2,
                'p_value': p_value,
                'degrees_of_freedom': dof,
                'significant': p_value < 0.05,
                'interpretation': self._interpret_chisquare(p_value)
            })
            
        elif test_type == 'anova':
            # For ANOVA, var1 must be categorical (with 3+ levels) and var2 numeric
            if not pd.api.types.is_numeric_dtype(valid_data[var2]):
                raise ValueError("The dependent variable must be numeric for ANOVA")
            
            # Get unique groups
            groups = valid_data[var1].unique()
            
            if len(groups) < 3:
                raise ValueError("The grouping variable must have at least 3 levels for ANOVA")
            
            # Prepare data for ANOVA
            group_data = []
            group_means = {}
            
            for group in groups:
                group_values = valid_data[valid_data[var1] == group][var2]
                group_data.append(group_values)
                group_means[str(group)] = group_values.mean()
            
            # Perform ANOVA
            f_stat, p_value = stats.f_oneway(*group_data)
            
            results.update({
                'groups': {str(group): len(valid_data[valid_data[var1] == group]) for group in groups},
                'means': group_means,
                'f_statistic': f_stat,
                'p_value': p_value,
                'significant': p_value < 0.05,
                'interpretation': self._interpret_anova(p_value)
            })
            
        else:
            raise ValueError(f"Test type '{test_type}' not supported")
        
        return results
    
    def _interpret_correlation(self, corr, p_value):
        """Interprets the results of a correlation."""
        if p_value >= 0.05:
            return "There is no evidence of significant correlation between the variables."
        
        strength = "weak"
        if abs(corr) >= 0.7:
            strength = "strong"
        elif abs(corr) >= 0.3:
            strength = "moderate"
            
        direction = "positive" if corr > 0 else "negative"
        
        return f"There is a {strength} {direction} correlation and statistically significant (p={p_value:.4f}) between the variables."
    
    def _interpret_ttest(self, p_value, mean1, mean2, group1, group2):
        """Interprets the results of a t-test."""
        if p_value >= 0.05:
            return f"There is no significant difference in the means between groups {group1} and {group2}."
        
        higher_group = group1 if mean1 > mean2 else group2
        lower_group = group2 if mean1 > mean2 else group1
        
        return f"There is a significant difference (p={p_value:.4f}) in the means between groups. The group {higher_group} has a significantly higher mean than the group {lower_group}."
    
    def _interpret_chisquare(self, p_value):
        """Interprets the results of a chi-square test."""
        if p_value >= 0.05:
            return "There is no evidence of association between the categorical variables."
        return f"There is a significant association (p={p_value:.4f}) between the categorical variables."
    
    def _interpret_anova(self, p_value):
        """Interprets the results of an ANOVA."""
        if p_value >= 0.05:
            return "There is no significant difference in the means between the groups."
        return f"There is at least one significant difference (p={p_value:.4f}) between the means of the groups."

    def perform_cluster_analysis(self, features=None, n_clusters=3):
        if self.data is None:
            raise ValueError("No data has been set. Use set_data() first.")
        
        # If no features specified, use all numeric columns
        if features is None:
            features = self.data.select_dtypes(include=['number']).columns.tolist()
        else:
            # Verify all features exist and are numeric
            for feature in features:
                if feature not in self.data.columns:
                    raise ValueError(f"Column '{feature}' not found in data")
                if not np.issubdtype(self.data[feature].dtype, np.number):
                    raise ValueError(f"Column '{feature}' is not numeric")
        
        # Prepare data for clustering
        X = self.data[features].copy()
        
        # Handle missing values
        X = X.fillna(X.mean())
        
        # Scale features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Perform clustering
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        clusters = kmeans.fit_predict(X_scaled)
        
        # Add cluster assignments to data
        cluster_data = self.data.copy()
        cluster_data['cluster'] = clusters
        
        # Calculate cluster statistics
        cluster_stats = {}
        for feature in features:
            cluster_stats[feature] = cluster_data.groupby('cluster')[feature].agg(['mean', 'std', 'count'])
        
        # Calculate cluster centers in original scale
        centers_scaled = kmeans.cluster_centers_
        centers_original = scaler.inverse_transform(centers_scaled)
        
        result = {
            'cluster_assignments': clusters,
            'cluster_statistics': cluster_stats,
            'cluster_centers': pd.DataFrame(centers_original, columns=features),
            'inertia': kmeans.inertia_
        }
        
        return result
    
    def analyze_performance_trends(self):
        if self.data is None:
            raise ValueError("No data has been set. Use set_data() first.")
        
        if 'performance_score' not in self.data.columns:
            raise ValueError("Data does not contain 'performance_score' column")
        
        result = {}
        
        # Performance distribution
        result['performance_stats'] = self.data['performance_score'].describe()
        
        # Performance by department if available
        if 'department' in self.data.columns:
            result['performance_by_dept'] = self.data.groupby('department')['performance_score'].agg(['mean', 'std', 'count'])
        
        # Performance by tenure if available
        if 'days_service' in self.data.columns:
            # Create tenure bins
            tenure_bins = [0, 365, 2*365, 5*365, float('inf')]
            tenure_labels = ['<1 year', '1-2 years', '2-5 years', '>5 years']
            self.data['tenure_group'] = pd.cut(self.data['days_service'], bins=tenure_bins, labels=tenure_labels)
            result['performance_by_tenure'] = self.data.groupby('tenure_group')['performance_score'].agg(['mean', 'std', 'count'])
        
        # Performance correlations with numeric variables
        numeric_cols = self.data.select_dtypes(include=['number']).columns
        perf_correlations = self.data[numeric_cols].corr()['performance_score'].drop('performance_score')
        result['performance_correlations'] = perf_correlations
        
        return result
    
    def analyze_education_salary_relationship(self):
        """
        Performs a comprehensive analysis of the relationship between education level and salary.
        
        Returns:
            dict: A dictionary with the results of the analysis, including:
                - statistics: Basic statistics of salary by education level
                - anova_test: Results of ANOVA test if there are more than 2 education levels
                - correlation: Results of correlation if education level can be mapped to numeric values
        """
        if self.data is None:
            raise ValueError("No data has been set. Use set_data() first.")
        
        if 'education' not in self.data.columns:
            raise ValueError("Data does not contain 'education' column")
        
        if 'base_salary' not in self.data.columns:
            raise ValueError("Data does not contain 'base_salary' column")
        
        results = {}
        
        # Get basic statistics by education level
        salary_factors = self.analyze_salary_factors()
        if 'salary_by_education' in salary_factors:
            results['statistics'] = salary_factors['salary_by_education']
        
        # ANOVA test for significant differences between levels
        education_levels = self.data['education'].unique()
        if len(education_levels) >= 3:
            try:
                anova_results = self.run_statistical_tests('education', 'base_salary', test_type='anova')
                results['anova_test'] = anova_results
            except Exception as e:
                results['anova_test_error'] = str(e)
        
        # Correlation analysis if we can map education to numeric values
        try:
            # Common education level mapping
            education_mapping = {
                'High School': 1, 'Secondary': 1, 'Secundaria': 1,
                'Associate': 2, 'Tecnico': 2, 'Technical': 2,
                'Bachelor': 3, 'Licenciatura': 3, 'Universitario': 3, 'College': 3,
                'Master': 4, 'Maestria': 4, 'Masters': 4,
                'PhD': 5, 'Doctorate': 5, 'Doctorado': 5
            }
            
            # Automatically detect levels and create custom mapping if needed
            detected_levels = self.data['education'].unique()
            customized_mapping = {}
            
            if not all(level in education_mapping for level in detected_levels):
                # If not all levels are in the predefined mapping, create a custom one
                sorted_levels = sorted(detected_levels, key=lambda x: str(x).lower())
                for i, level in enumerate(sorted_levels):
                    customized_mapping[level] = i + 1
                
                # Use the custom mapping
                education_mapping = customized_mapping
            
            # Create temporary copy with numeric mapping
            temp_df = self.data.copy()
            temp_df['education_numeric'] = temp_df['education'].map(education_mapping)
            
            # Set up temporary analyzer
            temp_analyzer = EmployeeDataAnalyzer(temp_df)
            
            # Perform correlation analysis
            corr_results = temp_analyzer.run_statistical_tests('education_numeric', 'base_salary', test_type='correlation')
            results['correlation'] = corr_results
            results['education_mapping'] = education_mapping
        except Exception as e:
            results['correlation_error'] = str(e)
        
        # Add comparative descriptive statistics
        try:
            results['comparative_stats'] = {}
            for level in self.data['education'].unique():
                level_salary = self.data[self.data['education'] == level]['base_salary']
                results['comparative_stats'][level] = {
                    'mean': level_salary.mean(),
                    'median': level_salary.median(),
                    'std': level_salary.std(),
                    'min': level_salary.min(),
                    'max': level_salary.max(),
                    'count': len(level_salary)
                }
        except Exception as e:
            results['comparative_stats_error'] = str(e)
        
        return results
