import pandas as pd
import seaborn as sbn
import matplotlib.pyplot as plt
from scipy import stats
from scipy.stats import mannwhitneyu
from scipy.stats import spearmanr 
from scipy.stats import pearsonr
from scipy.stats import chi2_contingency

# Load dataset
df = pd.read_excel(r"C:\Users\merie\OneDrive\Bureau\WGU\D599\Task2-D599\Health Insurance Dataset.xlsx")

## Part 1
# Continuous variable: Age
age_stats = df['age'].describe()
print(age_stats)

# Continuous variable: bmi
bmi_stats = df['bmi'].describe()
print(bmi_stats)

# Plot for Age
plt.subplot(2, 2, 1)
sbn.boxplot(x=df['age'])
plt.title('Boxplot of Age')
plt.xlabel('Age')

plt.subplot(2, 2, 2)
sbn.histplot(df['age'], bins=20, kde=True, color='skyblue')
plt.title('Histogram of Age')
plt.xlabel('Age')
plt.ylabel('Frequency')

# Plot for bmi
plt.subplot(2, 2, 3)
sbn.boxplot(x=df['bmi'])
plt.title('Boxplot of bmi')
plt.xlabel('bmi')

plt.subplot(2, 2, 4)
sbn.histplot(df['bmi'], bins=20, kde=True, color='salmon')
plt.title('Histogram of bmi')
plt.xlabel('bmi')
plt.ylabel('Frequency')

# Show the plots
plt.show()

# Categorical variables: sex & Smoker
# Summary statistics for sex variable 
sex_counts = df['sex'].value_counts()
sex_summary = pd.DataFrame(sex_counts) 
sex_summary.columns = ['Count']
sex_summary['Proportion'] = sex_summary['Count'] / sex_summary['Count'].sum() 

# Summary statistics for Smoker variable 
smoker_counts = df['smoker'].value_counts() 
smoker_summary = pd.DataFrame(smoker_counts) 
smoker_summary.columns = ['Count']
smoker_summary['Proportion'] = smoker_summary['Count'] / smoker_summary['Count'].sum()

# Print the summary statistics 
print("Summary Statistics for sex:") 
print(sex_summary)
print("\nSummary Statistics for Smoker:") 
print(smoker_summary)


# Plot for sex variable
plt.subplot(2, 2, 1)
sbn.countplot(data=df, x='sex', palette='pastel')
plt.title('Count Plot of sex')
plt.xlabel('sex')
plt.ylabel('Count')

# Plot for Smoker
plt.subplot(2, 2, 3)
sbn.countplot(data=df, x='smoker', palette='pastel')
plt.title('Count Plot of Smokers')
plt.xlabel('Smoker Status')
plt.ylabel('Count')

# Show the plots
plt.show()


#Scatter plot Age vs bmi
plt.figure(figsize=(8, 6)) 
sbn.scatterplot(x='age', y='bmi', data=df, color='skyblue')
plt.title('Scatter Plot of Age vs bmi') 
plt.xlabel('Age') 
plt.ylabel('bmi') 
plt.show()

#Pearson Correlation
correlation, p_value = pearsonr(df['age'], df['bmi'])
print(f'Pearson correlation coefficient: {correlation}')
print(f'p_value:{p_value}')

#Contingency table for sex vs Smoker 
contingency_table = pd.crosstab(df['sex'], df['smoker']) 
print("\nContingency Table (sex vs Smoker)")
print(contingency_table)

# Chi-square test 
chi_stat, p, dof, expected = chi2_contingency(contingency_table)
print(f"\nChi-square test statistic: {chi_stat:.2f}, P-value: {p:.5f}")


# Boxplot for bmi vs Smoker
plt.figure(figsize=(8, 8)) 
sbn.boxplot(x='bmi', y='smoker', data=df, palette='Set2') 
plt.title('Box Plot of bmi by smoker')
plt.xlabel('bmi') 
plt.ylabel('Smoker') 
plt.show()

#Statistics
print(df.groupby('smoker')['bmi'].describe())


## Part 2: Parametric Statistical Testing
# Check BMI variable
if len(df['bmi'].unique()) == 1:
    print("All BMI values are identical, Shapiro-Wilk cannot run.")
else:
    # Perform the Shapiro-Wilk test for normality on the BMI data
    shapiro_stat, p_value = stats.shapiro(df['bmi'])
    
    # Output the results
    print(f'Shapiro-Wilk Test Statistics={shapiro_stat}, p-value={p_value}')
    
    # Interpret the result
    if p_value <= 0.05:
        print("BMI does not appear to be normally distributed (p <= 0.05).")
    else:
        print("BMI appears to be normally distributed (p > 0.05).")

# Group the BMI values by patient levels
grouped_bmi = [group['bmi'].values for name, group in df.groupby('Level')]

# Perform ANOVA test
f_statistic, p_value_anova = stats.f_oneway(*grouped_bmi)

# Print ANOVA results
print(f'ANOVA F-statistic: {f_statistic}, p-value: {p_value_anova}')

# Interpret the ANOVA results
if p_value_anova < 0.05:
    print("There is a significant difference in BMI across different patient levels.")
else:
    print("There is no significant difference in BMI across different patient levels.")

# Part 3: Nonparametric Statistical Testing

# Ensure 'charges' column is numeric
df['charges'] = pd.to_numeric(df['charges'], errors='coerce')

# Perform the Mann-Whitney U test
# Filter the data by gender
male_charges = df[df['sex'] == 'male']['charges']
female_charges = df[df['sex'] == 'female']['charges']

# Perform the Mann-Whitney U test
charges_stat, charges_p_value = stats.mannwhitneyu(male_charges, female_charges)

# Print results
print(f"Mann-Whitney U test statistic: {charges_stat}, p-value: {charges_p_value}")

# Interpretation of the results
if charges_p_value < 0.05:
    print("There is a significant difference in insurance charges between male and female policyholders.")
else:
    print("There is no significant difference in insurance charges between male and female policyholders.")

# Create a boxplot to visualize the distribution of insurance charges by sex
plt.figure(figsize=(10, 6))
sbn.boxplot(x='sex', y='charges', data=df)
plt.title('Distribution of Insurance Charges by sex')
plt.xlabel('sex')
plt.ylabel('Insurance Charges')
plt.show()

# Create histograms to compare the distribution of charges by sex
plt.figure(figsize=(10, 6))
sbn.histplot(data=df, x='charges', y='sex', kde=True, bins=30)
plt.title('Insurance Charges Distribution by sex')
plt.xlabel('Insurance Charges')
plt.ylabel('Frequency')
plt.show()
