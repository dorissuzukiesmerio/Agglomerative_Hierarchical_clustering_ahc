import pandas

import matplotlib.pyplot as pyplot

import scipy.cluster.hierarchy as shc # scientific python , easier to operate than numpy 

dataset = pandas.read_csv("dataset_ahc.csv")
print(dataset)

pyplot.scatter(dataset['x1'],dataset['x2']) # obs: or transform to dataset.values and use numpy syntax
pyplot.savefig("scatterplot_ahc.png")
pyplot.close()

pyplot.title("Dendrogram")
dendrogram_object = shc.dendrogram(shc.linkage(dataset, method="ward")) # how the python is the distance , hence what we mean by closest cluster
pyplot.savefig("dendrogram.png")
pyplot.close()

# Obs: different methods:
# sum of squares distances  'ward'
# complete : farthest 
# closest 

machine = AgglomerativeClustering(n_clusters=4, affinity="euclidean", linkage="ward")
results_ahc = machine.fit_predict(dataset)

pyplot.scatter(dataset['x1'],dataset['x2']) # obs: or transform to dataset.values and use numpy syntax
pyplot.savefig("scatterplot_ahc.png")
pyplot.close()

#Once you cluster, you cannot "decluster" as improving 
