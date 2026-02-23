import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.cluster import KMeans

sns.set(style="darkgrid")
plt.rcParams["figure.figsize"] = (8, 6)

print("\n CUSTOMER SEGMENTATION USING K-MEANS \n")

print(" Loading dataset...")
data = pd.read_csv("Mall_Customers.csv")

print("\n Dataset Preview:")
print(data.head())

X = data[["Annual Income (k$)", "Spending Score (1-100)"]]

print("\n Selected Features:")
print(X.head())

kmeans = KMeans(n_clusters=5, random_state=42)
clusters = kmeans.fit_predict(X)

data["Cluster"] = clusters

print("\n K-Means clustering completed!")

print("\n Displaying Customer Segments...")

plt.figure()
sns.scatterplot(
    x="Annual Income (k$)",
    y="Spending Score (1-100)",
    hue="Cluster",
    palette="rainbow",
    data=data,
    s=80
)

plt.scatter(
    kmeans.cluster_centers_[:, 0],
    kmeans.cluster_centers_[:, 1],
    color="black",
    marker="X",
    s=200,
    label="Centroids"
)

plt.title("Customer Segmentation using K-Means ")
plt.xlabel("Annual Income (k$)")
plt.ylabel("Spending Score (1-100)")
plt.legend()
plt.show()

print("\n END OF PROGRAM ")
