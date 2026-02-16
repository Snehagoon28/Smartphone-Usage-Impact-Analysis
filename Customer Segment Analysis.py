import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error
from sklearn.preprocessing import LabelEncoder, StandardScaler

# =================================================================
# PHASE 1: ENTERPRISE ETL (Extract, Transform, Load)
# =================================================================
def phase_1_etl(file_path):
    print("\n" + "="*60)
    print("PHASE 1: ADVANCED ETL PIPELINE STARTING...")
    print("="*60)
    
    # 1.1 Data Extraction
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Critical Error: File not found at {file_path}")
    
    df = pd.read_csv(file_path)
    
    # 1.2 Data Cleaning
    initial_count = len(df)
    df = df.drop_duplicates()
    df = df.dropna()
    print(f"Log: Removed {initial_count - len(df)} duplicate/null rows.")
    
    # 1.3 Advanced Feature Engineering
    print("Log: Engineering complex features...")
    
    # Digital Exhaustion Index: Ratio of Phone Usage to Sleep
    df['Exhaustion_Index'] = df['Daily_Phone_Hours'] / (df['Sleep_Hours'] + 1)
    
    # Productivity Efficiency: Score per hour of phone use
    df['Prod_Per_Hour'] = df['Work_Productivity_Score'] / (df['Daily_Phone_Hours'] + 0.1)
    
    # Weekend vs Weekday Intensity
    df['Weekend_Intensity'] = df['Weekend_Screen_Time_Hours'] / (df['Daily_Phone_Hours'] + 0.1)
    
    # Age Demographics
    df['Age_Bracket'] = pd.cut(df['Age'], bins=[0, 25, 40, 60, 100], 
                               labels=['Gen Z', 'Millennials', 'Gen X', 'Seniors'])
    
    print(f"Success: ETL Finished. Final Dataset Columns: {list(df.columns)}")
    return df

# =================================================================
# PHASE 2: MULTI-DIMENSIONAL ANALYSIS
# =================================================================
def phase_2_analysis(df):
    print("\n" + "="*60)
    print("PHASE 2: DEEP STATISTICAL PROFILING...")
    print("="*60)
    
    # 2.1 Grouped Metrics
    report = df.groupby('Occupation').agg({
        'Stress_Level': ['mean', 'std'],
        'Work_Productivity_Score': 'mean',
        'Exhaustion_Index': 'mean'
    }).round(2)
    
    print("\n[Occupational Stress & Productivity Profile]")
    print(report)
    
    # 2.2 Correlation Analysis
    numeric_df = df.select_dtypes(include=[np.number])
    corr_stress = numeric_df.corr()['Stress_Level'].sort_values(ascending=False)
    
    print("\n[Top Statistical Stress Correlates]")
    print(corr_stress.head(5))
    
    # 2.3 Habit Benchmarking
    print("\n[Global Habit Averages]")
    print(f"• Avg Screen Time: {df['Daily_Phone_Hours'].mean():.2f} hrs")
    print(f"• Avg Sleep:       {df['Sleep_Hours'].mean():.2f} hrs")
    print(f"• Avg Caffeine:    {df['Caffeine_Intake_Cups'].mean():.2f} cups")

# =================================================================
# PHASE 3: MASTER VISUALIZATION SUITE (8 CHARTS)
# =================================================================
def phase_3_visuals(df):
    print("\n" + "="*60)
    print("PHASE 3: GENERATING HIGH-IMPACT VISUALIZATIONS...")
    print("="*60)
    
    # Set global style
    sns.set_theme(style="whitegrid", palette="muted")
    
    # 1. MASTER HEATMAP
    plt.figure(figsize=(14, 10))
    sns.heatmap(df.select_dtypes(include=[np.number]).corr(), annot=True, cmap='coolwarm', center=0)
    plt.title('1. Inter-Variable Correlation Heatmap', fontsize=16, pad=20)
    plt.show()

    # 2. DENSITY MAPPING: USAGE VS PRODUCTIVITY
    plt.figure(figsize=(10, 7))
    plt.hexbin(df['Daily_Phone_Hours'], df['Work_Productivity_Score'], gridsize=35, cmap='YlGnBu')
    plt.colorbar(label='Count of Users')
    plt.xlabel('Daily Phone Hours')
    plt.ylabel('Productivity Score')
    plt.title('2. Population Density: Phone Usage vs Efficiency', fontsize=14)
    plt.show()

    # 3. DISTRIBUTION: STRESS BY GENDER (VIOLIN)
    plt.figure(figsize=(10, 6))
    sns.violinplot(data=df, x='Gender', y='Stress_Level', hue='Device_Type', split=True, palette='Set2')
    plt.title('3. Stress Level Density by Gender & Device', fontsize=14)
    plt.show()

    # 4. TREND ANALYSIS: SLEEP VS STRESS (REGRESSION)
    plt.figure(figsize=(10, 6))
    sns.regplot(data=df.sample(1000), x='Sleep_Hours', y='Stress_Level', 
                scatter_kws={'alpha':0.2}, line_kws={'color':'red', 'lw':3})
    plt.title('4. Statistical Trend: Impact of Sleep Deficiency on Stress', fontsize=14)
    plt.show()

    # 5. CATEGORICAL ANALYSIS: PRODUCTIVITY BY OCCUPATION
    plt.figure(figsize=(12, 6))
    sns.boxenplot(data=df, x='Occupation', y='Work_Productivity_Score', palette='magma')
    plt.title('5. Distribution of Productivity Across Career Paths', fontsize=14)
    plt.show()

    # 6. DIGITAL EXHAUSTION ACROSS GENERATIONS
    plt.figure(figsize=(10, 6))
    sns.barplot(data=df, x='Age_Bracket', y='Exhaustion_Index', palette='viridis', ci='sd')
    plt.title('6. Mean Exhaustion Index by Generation (Error Bars = Std Dev)', fontsize=14)
    plt.show()

    # 7. MULTIVARIATE: CAFFEINE VS STRESS VS PRODUCTIVITY
    plt.figure(figsize=(11, 7))
    sns.scatterplot(data=df.sample(1000), x='Caffeine_Intake_Cups', y='Stress_Level', 
                    hue='Work_Productivity_Score', size='Daily_Phone_Hours', alpha=0.7, palette='Spectral')
    plt.title('7. Multivariate Bubble Chart: Stress Drivers', fontsize=14)
    plt.legend(bbox_to_anchor=(1.05, 1), loc=2)
    plt.tight_layout()
    plt.show()

    # 8. KERNEL DENSITY: SOCIAL MEDIA BY DEVICE
    plt.figure(figsize=(10, 6))
    sns.kdeplot(data=df, x='Social_Media_Hours', hue='Device_Type', fill=True, alpha=0.5)
    plt.title('8. Probability Distribution: Social Media Consumption by Platform', fontsize=14)
    plt.show()

# =================================================================
# PHASE 4: MACHINE LEARNING & PREDICTION
# =================================================================
def phase_4_ml(df):
    print("\n" + "="*60)
    print("PHASE 4: PREDICTIVE MODELING & AI INSIGHTS...")
    print("="*60)
    
    # 4.1 Encoding and Selection
    ml_df = df.copy().drop(columns=['User_ID', 'Age_Bracket'])
    le = LabelEncoder()
    for col in ml_df.select_dtypes(include=['object']).columns:
        ml_df[col] = le.fit_transform(ml_df[col])
    
    X = ml_df.drop('Stress_Level', axis=1)
    y = ml_df['Stress_Level']
    
    # 4.2 Training
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    model = RandomForestRegressor(n_estimators=100, max_depth=12, n_jobs=-1, random_state=42)
    model.fit(X_train, y_train)
    
    # 4.3 Evaluation
    preds = model.predict(X_test)
    print(f"• Model Performance (R2 Score): {r2_score(y_test, preds):.4f}")
    print(f"• Mean Absolute Error:           {mean_absolute_error(y_test, preds):.4f}")
    
    # 4.4 Feature Importance Visual
    plt.figure(figsize=(10, 6))
    importances = pd.Series(model.feature_importances_, index=X.columns).sort_values()
    importances.plot(kind='barh', color='teal')
    plt.title('9. AI Insight: Key Mathematical Predictors of Stress', fontsize=14)
    plt.show()

# =================================================================
# EXECUTION ENTRY POINT
# =================================================================
if __name__ == "__main__":
    # YOUR MAC FILE PATH
    SNEHA_PATH = '/Users/sneha/Documents/Projects/1/Smartphone_Usage_Productivity_Dataset_50000.csv'
    
    try:
        processed_data = phase_1_etl(SNEHA_PATH)
        phase_2_analysis(processed_data)
        phase_3_visuals(processed_data)
        phase_4_ml(processed_data)
        print("\n[SUCCESS] ANALYSIS PIPELINE COMPLETED SUCCESSFULLY.")
    except Exception as e:
        print(f"\n[CRITICAL ERROR] Pipeline failed: {e}")