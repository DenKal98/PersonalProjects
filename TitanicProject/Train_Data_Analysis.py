import pandas as pd 
import seaborn as sns
from matplotlib import pyplot as plt 



df  = pd.read_csv("data/train.csv")




#Need to analyse which data would be most relevant for the ML Model
sur = "Survived"
not_sur = "Not Survived"
new_df = df.head(5)

#Creating a plot to see the different sex and age of the survivors compared to non survivors
fig, axes = plt.subplots(nrows=1, ncols=2,figsize=(10,4))
men = df[df["Sex"]== "male"]
female = df[df["Sex"] == "female"]
women_sur = female[female["Survived"] == 1]
women_not_sur = female[female["Survived"] == 0]
men_sur = men[men["Survived"] == 1]
men_not_sur = men[men["Survived"] == 0]

'''MALE'''
ax = sns.distplot(men_sur.Age.dropna(), bins=18, label = sur, color= "g", ax = axes[1], kde =False)
ax = sns.distplot(men_not_sur.Age.dropna(), bins=40, label = not_sur, color="r",  ax = axes[1], kde =False)
ax.legend()
ax.set_title("Male")

'''FEMALE'''
ax = sns.distplot(women_sur.Age.dropna(),bins=18, label = sur, color= "g", ax = axes[0], kde =False)
ax = sns.distplot(women_not_sur.Age.dropna(), bins=40, label = not_sur, color="r",  ax = axes[0], kde =False)
ax.legend()
ax.set_title("Female")


#Comparing different ticket classes to survival by where they embarked their journey 
FacetGrid = sns.FacetGrid(df, row='Embarked', size=3, aspect=1.6)
FacetGrid.map(sns.pointplot, 'Pclass', 'Survived', 'Sex', palette=None,  order=None, hue_order=None )
FacetGrid.add_legend()

#PClass in comparison to Survival
#sns.barplot(x='Pclass', y='Survived', data=df)
sns.countplot(x='Survived', hue='Pclass', data=df)



