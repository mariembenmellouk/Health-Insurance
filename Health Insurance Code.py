import pandas as pd
import seaborn as sbn
import matplotlib.pyplot as plt
from scipy import stats
from scipy.stats import spearmanr 

# Load dataset
df = pd.read_excel(r"C:\WGU\D599\Task 2\Health Insurance Dataset.xlsx")

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

# Contingency table for sex vs Smoker 
contingency_table = pd.crosstab(df['sex'], df['smoker']) 
print("\nContingency Table (sex vs Smoker)")
print(contingency_table)

# Boxplot for bmi vs Smoker
plt.figure(figsize=(8, 8)) 
sbn.boxplot(x='bmi', y='smoker', data=df, palette='Set2') 
plt.title('Box Plot of bmi by smoker')
plt.xlabel('bmi') 
plt.ylabel('Smoker') 
plt.show()

## Part 2: Parametric Statistical Testing

# Check data type of all columns
print(df.dtypes)

# Convert charges to numeric, coercing errors to NaN
df['charges'] = pd.to_numeric(df['charges'], errors='coerce')
# Verify the conversion
print("Data type of charges after conversion:", df['charges'].dtype)

# Create two groups based on smoking status
smokers = df[df['smoker'] == 'yes']['charges']
non_smokers = df[df['smoker'] == 'no']['charges']

# Perform the independent samples t-test
t_statistic, p_value = stats.ttest_ind(smokers, non_smokers, equal_var=False)

# Print results 
print(f"T-statistic: {t_statistic}") 
print(f"P-value: {p_value}")

if p_value < 0.05:
    print("There is a significant difference in insurance charges between smokers and non-smokers.")
else:
    print("There is no significant difference in insurance charges between smokers and non-smokers.")

# Part 3: Nonparametric Statistical Testing

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