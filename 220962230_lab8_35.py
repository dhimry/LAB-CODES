#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np 
from matplotlib import pyplot as plt


# In[4]:


df=pd.DataFrame({'FruitId':[1,2,3,4,5,6] , 'Weight': [180,200,150,170,160,140] , 'Sweetness': [7,6,4,5,6,3] , 
                 'Label': ['Apple', 'Apple', 'Orange' , 'Orange','Apple','Orange']})
df


# In[5]:


# df.to_csv('Fruits.csv')


# In[35]:


import numpy as np

def euclidean_distance(point1, point2):
    return np.sqrt(np.sum((point1 - point2) ** 2))

def manhattan_distance(point1, point2):
    return np.sum(np.abs(point1 - point2))

def minkowski_distance(point1, point2, p):
    return np.sum(np.abs(point1 - point2) ** p) ** (1/p)
def knn_classifier(new_fruit, dataset, k,printdist):
    distances = []
    
    for fruit in zip(dataset['FruitId'],dataset['Weight'] , dataset['Sweetness'] , dataset['Label']):
        
        fruitid,weight, sweetness, label = fruit
        point = np.array([weight, sweetness])
        dist_euclidean = euclidean_distance(new_fruit, point)
        dist_manhattan = manhattan_distance(new_fruit, point)
        dist_minkowski = minkowski_distance(new_fruit, point, p=3)  # For p=3

        distances.append((dist_euclidean, dist_manhattan, dist_minkowski, label,fruitid))

    if printdist==True:
        print('Distances: ')
        df2=pd.DataFrame({'FruitId':[x[4] for x in distances],'Euclidian': [x[0] for x in distances],'Manhattan': [x[1] for x in distances],
                     'Minkowski': [x[2] for x in distances] , 'Labels':[x[3] for x in distances]})
        print(df2)
    distances.sort(key=lambda x: x[0])
    neighbors = distances[:k]
    
    from collections import Counter
    labels = [neighbor[3] for neighbor in neighbors]
    most_common = Counter(labels).most_common(1)
    
    return most_common[0][0], neighbors  # Return predicted label and neighbor distances
new_fruit = (165, 5.5)
print(new_fruit)
predicted_label, neighbors = knn_classifier(new_fruit, df, k=3,printdist=True)
print("Predicted Label:", predicted_label)



# In[37]:


print('for k=1:')
predicted_label, neighbors = knn_classifier(new_fruit, df, k=1,printdist=False)
print(predicted_label)
print('for k=5:')
predicted_label, neighbors = knn_classifier(new_fruit, df, k=5,printdist=False)
print(predicted_label)


# In[43]:


# Visualization
weights = df['Weight'].to_numpy()
sweetness = df['Sweetness'].to_numpy()
labels = df['Label'].to_numpy()

# Map labels to numerical values for contour plotting
label_map = {'Apple': 0, 'Orange': 1}
Z = np.array([label_map[label] for label in labels])

plt.scatter(weights, sweetness, color=['red' if label == 'Apple' else 'orange' for label in labels])
plt.scatter(new_fruit[0], new_fruit[1], color='blue', marker='x', s=100)

# Create a mesh grid for decision boundary
x_min, x_max = weights.min() - 10, weights.max() + 10
y_min, y_max = sweetness.min() - 1, sweetness.max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, 1), np.arange(y_min, y_max, 0.1))

# Classify each point in the mesh grid
Z_grid = np.array([knn_classifier((x, y), df, k=3,printdist=False)[0] for x, y in zip(np.ravel(xx), np.ravel(yy))])
Z_grid_numeric = np.array([label_map[label] for label in Z_grid]).reshape(xx.shape)

# Plot decision boundary
plt.contourf(xx, yy, Z_grid_numeric, alpha=0.3, levels=np.arange(-0.5, 2, 1), colors=['red', 'orange'], extend='both')

# Set limits and labels
plt.xlim(x_min, x_max)
plt.ylim(y_min, y_max)
plt.title("KNN Fruit Classification with Decision Boundary")
plt.xlabel("Weight (grams)")
plt.ylabel("Sweetness Level")
plt.grid(True)
plt.show()


# In[47]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier

# Your provided DataFrame
df = pd.DataFrame({
    'FruitId': [1, 2, 3, 4, 5, 6],
    'Weight': [180, 200, 150, 170, 160, 140],
    'Sweetness': [7, 6, 4, 5, 6, 3],
    'Label': ['Apple', 'Apple', 'Orange', 'Orange', 'Apple', 'Orange']
})

# Features and Labels
X = df[['Weight', 'Sweetness']]
y = df['Label']

# Instantiate and fit the KNN model
knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X, y)

# New fruit characteristics
new_fruit = pd.DataFrame([[165, 5.5]], columns=['Weight', 'Sweetness'])

# Classify the new fruit
predicted_label = knn.predict(new_fruit)
print("Predicted Label:", predicted_label[0])

# Visualization
weights = df['Weight'].to_numpy()
sweetness = df['Sweetness'].to_numpy()

# Map labels to colors
label_color_map = {'Apple': 'red', 'Orange': 'orange'}
colors = [label_color_map[label] for label in y]

plt.scatter(weights, sweetness, color=colors)
plt.scatter(new_fruit.iloc[0]['Weight'], new_fruit.iloc[0]['Sweetness'], color='blue', marker='x', s=100)

# Create a mesh grid for decision boundary
x_min, x_max = weights.min() - 10, weights.max() + 10
y_min, y_max = sweetness.min() - 1, sweetness.max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, 1), np.arange(y_min, y_max, 0.1))

# Create a DataFrame for mesh grid points
mesh_points = pd.DataFrame(np.c_[xx.ravel(), yy.ravel()], columns=['Weight', 'Sweetness'])

# Classify each point in the mesh grid
Z = knn.predict(mesh_points)

# Create a color map for the decision boundary
Z_numeric = np.array([1 if label == 'Apple' else 0 for label in Z]).reshape(xx.shape)

# Plot decision boundary
plt.contourf(xx, yy, Z_numeric, alpha=0.3, colors=['orange', 'red'], levels=[-0.5, 0.5, 1.5])

# Set limits and labels
plt.xlim(x_min, x_max)
plt.ylim(y_min, y_max)
plt.title("KNN Fruit Classification with Decision Boundary (Scikit-Learn)")
plt.xlabel("Weight (grams)")
plt.ylabel("Sweetness Level")
plt.grid(True)
plt.show()


# In[61]:


import numpy as np
import pandas as pd
from collections import Counter
from math import log2

# Sample dataset
data = {
    'ID': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    'Age': [30, 45, 50, 35, 60, 55, 40, 25, 65, 45],
    'Blood Pressure': ['High', 'Low', 'High', 'Low', 'High', 'Low', 'High', 'Low', 'High', 'Low'],
    'Cholesterol': ['High', 'Normal', 'High', 'Normal', 'High', 'Normal', 'High', 'Normal', 'High', 'Normal'],
    'Diagnosis': ['Sick', 'Healthy', 'Sick', 'Healthy', 'Sick', 'Healthy', 'Sick', 'Healthy', 'Sick', 'Healthy']
}
# print(data)
df = pd.DataFrame(data)
print(df)
# Step 1: Calculate Entropy for the target variable


df = pd.DataFrame(data)

# Function to calculate entropy
def entropy(target_col):
    elements, counts = np.unique(target_col, return_counts=True)
    entropy_value = 0
    total_count = sum(counts)
    
    for count in counts:
        p = count / total_count
        entropy_value -= p * log2(p)
    
    return entropy_value

# Function to calculate information gain
def information_gain_age(data, target):
    total_entropy = entropy(data[target])
    weighted_entropy = 0
    
    # Calculate the unique ages
    age_values = np.unique(data['Age'])
    
    for age in age_values:
        subset = data[data['Age'] == age]
        subset_entropy = entropy(subset[target])
        weighted_entropy += (len(subset) / len(data)) * subset_entropy
    
    return total_entropy - weighted_entropy

# Decision Tree Class with Splitting Information
class DecisionTree:
    def __init__(self, data, target):
        self.target = target
        self.tree = self.build_tree(data)
    
    def build_tree(self, data):
        target_values = data[self.target].unique()
        
        # If all target values are the same, return that value
        if len(target_values) == 1:
            return target_values[0]
        
        # If there are no features left to split, return the majority class
        if len(data.columns) == 1:
            return data[self.target].mode()[0]
        
        # Find the best feature to split on
        best_feature = max(data.columns.difference([self.target]), key=lambda feature: information_gain(data, feature, self.target))
        tree = {best_feature: {}}
        
        print(f"\nSplitting on feature: {best_feature}")
        
        for value in data[best_feature].unique():
            subset = data[data[best_feature] == value].drop(columns=[best_feature])
            print(f" - Value: {value}, Subset size: {len(subset)}")
            tree[best_feature][value] = self.build_tree(subset)
        
        return tree

    def predict(self, sample):
        node = self.tree
        while isinstance(node, dict):
            feature = next(iter(node))
            value = sample[feature]
            node = node[feature].get(value, None)
        return node

# Build the decision tree
# dtree = DecisionTree(df, 'Diagnosis')
diagnosis_entropy = entropy(df['Diagnosis'])
print(f'Entropy for Diagnosis: {diagnosis_entropy}')
features = ['Age', 'Blood Pressure', 'Cholesterol']
info_gains = {feature: information_gain(df, feature, 'Diagnosis') for feature in features}

# Display information gains
for feature, gain in info_gains.items():
    print(f'Information Gain for {feature}: {gain}')
# Step 5: Predict
sample = {'Age': 50, 'Blood Pressure': 'Low', 'Cholesterol': 'Normal'}
# prediction = dtree.predict(sample)


info_gains = {feature: information_gain(df, feature, 'Diagnosis') for feature in features}

# Identify the feature with the highest information gain
best_feature = max(info_gains, key=info_gains.get)

# Display the information gains and the best feature
print("Information Gains:")
for feature, gain in info_gains.items():
    print(f"{feature}: {gain}")

print(f"\nThe feature chosen as the root node is: {best_feature}")


# print(f'The prediction for the patient is: {prediction}')


# In[62]:


dtree = DecisionTree(df, 'Diagnosis')

# Display the tree structure
import pprint
pprint.pprint(dtree.tree)

# Example prediction
sample = {'Age': 50, 'Blood Pressure': 'Low', 'Cholesterol': 'Normal'}
prediction = dtree.predict(sample)
print(f'Prediction for the sample: {prediction}')


# In[64]:


import pandas as pd
from sklearn.tree import DecisionTreeClassifier, export_text

# Create the DataFrame
data = {
    'ID': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    'Age': [30, 45, 50, 35, 60, 55, 40, 25, 65, 45],
    'Blood Pressure': ['High', 'Low', 'High', 'Low', 'High', 'Low', 'High', 'Low', 'High', 'Low'],
    'Cholesterol': ['High', 'Normal', 'High', 'Normal', 'High', 'Normal', 'High', 'Normal', 'High', 'Normal'],
    'Diagnosis': ['Sick', 'Healthy', 'Sick', 'Healthy', 'Sick', 'Healthy', 'Sick', 'Healthy', 'Sick', 'Healthy']
}

df = pd.DataFrame(data)

# Encode categorical features
df_encoded = pd.get_dummies(df, columns=['Blood Pressure', 'Cholesterol'], drop_first=True)

# Features and target
X = df_encoded.drop(columns=['ID', 'Diagnosis'])
y = df_encoded['Diagnosis']

# Initialize and fit the Decision Tree Classifier
clf = DecisionTreeClassifier(criterion='entropy')
clf.fit(X, y)

# Print the feature importances and decision tree structure
print("Feature importances:", clf.feature_importances_)
print("Root feature:", X.columns[clf.tree_.feature[0]])
tree_rules = export_text(clf, feature_names=list(X.columns))
print("\nDecision Tree Rules:\n", tree_rules)

# Prepare a sample patient with consistent encoding
sample_patient = pd.DataFrame({
    'Age': [50],
    'Blood Pressure_High': [0],  # Low
    'Blood Pressure_Low': [1],    # Low
    'Cholesterol_High': [0],       # Normal
    'Cholesterol_Normal': [1]      # Normal
})

# Ensure sample_patient has the same columns as X
sample_patient = sample_patient.reindex(columns=X.columns, fill_value=0)

# Predict for the specific patient
prediction = clf.predict(sample_patient)
print(f"\nPrediction for a 50-year-old patient with low blood pressure and normal cholesterol: {prediction[0]}")


# In[ ]:




