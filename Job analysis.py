import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import re

# Load data
df = pd.read_csv("job_postings.csv", encoding='latin1')
print(df.head())
print(df.info())

# Clean data
df["job_title"] = df["job_title"].str.strip().str.lower()
df["location"] = df["location"].str.strip().str.lower()
df["company_name"] = df["company_name"].str.strip()
df = df.dropna(subset=['job_title', 'location'])

# Extract skills
skill_keywords = ['python', 'sql', 'excel', 'tableau', 'java', 'aws', 'react', 'docker', 'tensorflow', 'pandas', 'spark', 'power bi']
def infer_skills_from_title(title):
    return [skill for skill in skill_keywords if skill in title]

df['skills'] = df['job_title'].apply(infer_skills_from_title)
df_exploded = df.explode('skills')

# Clean weird characters
def clean_text(text):
    if isinstance(text, list):  # handle list input
        return [re.sub(r'[^\x00-\x7F]+', ' ', str(t)).strip().lower() for t in text]
    if pd.isna(text):
        return ""
    return re.sub(r'[^\x00-\x7F]+', ' ', str(text)).strip().lower()

df['job_title'] = df['job_title'].apply(clean_text)
df_exploded['job_title'] = df_exploded['job_title'].apply(clean_text)
df_exploded['skills'] = df_exploded['skills'].apply(clean_text)

# Create skill vs role matrix BEFORE using it
skill_role_matrix = df_exploded.groupby(['job_title', 'skills']).size().unstack(fill_value=0)
skill_role_matrix.to_excel("skill_vs_role_matrix.xlsx")

# Job demand analysis
job_demand = df['job_title'].value_counts()
job_demand.head(20).to_excel("top_job_titles.xlsx")

# Top 5 job roles
print("Top 5 In-Demand Roles:")
print(job_demand.head(5))

# Skill recommendation for top roles
recommended_roles = job_demand.head(5).index.tolist()
for role in recommended_roles:
    if role in skill_role_matrix.index:
        top_skills = skill_role_matrix.loc[role].sort_values(ascending=False).head(5)
        print(f"\nRecommended skills for '{role.title()}':")
        print(", ".join(top_skills.index))
    else:
        print(f"\nNo skills data found for role: {role}")

# Heatmap of top skills by top cities
top_cities = df["location"].value_counts().head(5).index
top_skills = df_exploded['skills'].value_counts().head(10).index
filtered = df_exploded[df_exploded["location"].isin(top_cities) & df_exploded['skills'].isin(top_skills)]
heatmap_data = filtered.groupby(['skills', 'location']).size().unstack(fill_value=0)

plt.figure(figsize=(10, 6))
sns.heatmap(heatmap_data, annot=True, cmap="YlGnBu")
plt.title("Top 10 Skills by City")
plt.tight_layout()
plt.show()

print("✅ Analysis completed. Excel files saved.")
