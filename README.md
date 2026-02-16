# Smartphone-Usage-Impact-Analysis

"How does our digital lifestyle affect our biological wellbeing and professional output?"

In an era of hyper-connectivity, we often ignore the hidden costs of smartphone usage. This project analyzes 50,000 data points to solve three specific problems:

- The Stress Correlation: Identifying which digital habits (screen time, app count) are the primary drivers of psychological stress.
- The Productivity Paradox: Determining if high smartphone usage actually aids work efficiency or creates a "diminishing returns" effect.
- Lifestyle Optimization: Providing a data-driven blueprint for the ideal balance between sleep, caffeine, and digital engagement.

Detailed Explanation of VisualizationsThe code generates 8+ "technically strong" charts. Here is the logic behind each:

1. Correlation Heatmap:What it shows: A grid of color-coded numbers representing the "Pearson Correlation Coefficient" between all variables.Why it’s used: It is the "first look" at the data. It tells us if $X$ goes up, does $Y$ go up? (e.g., as Daily Phone Hours increase, does Stress Level also increase?).
2. Hexbin Density Plot (Usage vs. Productivity):What it shows: Instead of messy dots, it uses hexagonal bins. Darker colors indicate a higher concentration of users.Why it’s used: With 50,000 records, a standard scatter plot would be a solid block of color. Hexbinning allows us to see the "center of gravity" of the population.
3. Violin Plot (Stress by Occupation):What it shows: The "fatness" of the violin shows the density of people at that stress level.Why it’s used: Unlike a bar chart which only shows the average, a violin plot shows the distribution. It reveals if a specific job has a wide range of stress or if everyone is equally stressed.
4. Regression Plot (Sleep vs. Stress):What it shows: Individual data points with a solid red line cutting through them.Why it’s used: This is a statistical tool to prove a trend. The red line represents the "Best Fit." If the line slopes down, it mathematically proves that more sleep leads to lower stress.
5. KDE Plot (Social Media by Gender):What it shows: A smooth, "hill-like" curve representing the probability of a user spending $X$ hours on social media.Why it’s used: It allows for a clean comparison between categories (Male vs. Female) without the clunkiness of histogram bars.
6. Boxen Plot (Productivity by Generation):What it shows: An "enhanced" boxplot with multiple levels of boxes.Why it’s used: Standard boxplots only show the 25th, 50th, and 75th percentiles. Boxen plots show even more "slices" of the data, which is necessary for large datasets to see the behavior of outliers.
7. Joint Plot (Caffeine vs. Productivity):What it shows: A scatter plot in the middle with histograms on the top and right sides.Why it’s used: It shows the relationship between two variables and the distribution of each variable individually, all in one view.
8. Line Plot (Age vs. App Usage):What it shows: A continuous line tracking how app count changes as users get older.Why it’s used: To identify "Digital Life Stages"—checking if younger users are more "app-heavy" than older professionals.

Explanation of the OutputWhen you run the code in VS Code, you will see two types of output:Terminal Output 
1. Data Integrity: Confirmation that 50,000 rows were processed.Statistical Report: A table showing which occupation (e.g., Student) is the most stressed.
2. ML Accuracy: An $R^2$ score (Coefficient of Determination). An $R^2$ closer to $1.0$ means the AI is very accurate at predicting stress.Pop-up Windows
3. (Visuals):You will see the 8 charts discussed above. These are the visual "proof" of your analysis.

Summary for Presentation -
"My project takes 50,000 user records and cleans them using a professional ETL pipeline. I used Advanced Feature Engineering to create a 'Digital Strain Index.' I then generated 8 high-density visualizations to prove that sleep and screen time are the biggest drivers of stress. Finally, I trained a Random Forest AI model that can predict a person's stress level based on their smartphone habits with high mathematical accuracy."
