from math import sqrt
 
# calculate the Euclidean distance between two vectors
def euclidean_distance(row1, row2):
	distance = 0.0
	for i in range(len(row1)):
		distance += (row1[i] - row2[i])**2
	return sqrt(distance)
 
# Locate the most similar neighbors
def get_neighbors(train, test_row, num_neighbors):
	distances = list()
	for train_row in train:
		dist = euclidean_distance(test_row, train_row)
		distances.append((train_row, dist))
	distances.sort(key=lambda tup: tup[1])
	neighbors = list()
	for i in range(num_neighbors):
		neighbors.append(distances[i][0])
	return neighbors
 
# Make a classification prediction with neighbors
def predict_classification(train, test_row, num_neighbors,test_val):
	neighbors = get_neighbors(train, test_row, num_neighbors)
	output_values = [row[-1] for row in neighbors]
	# print(output_values)
	can_use = all(val == test_val for val in output_values)
	# prediction = max(set(output_values), key=output_values.count)
	return can_use
 


def main():

	# Test distance function
	dataset = [[2.7810836,2.550537003,0],
		[1.465489372,2.362125076,0],
		[3.396561688,4.400293529,0],
		[1.38807019,1.850220317,0],
		[3.06407232,3.005305973,0],
		[7.627531214,2.759262235,1],
		[5.332441248,2.088626775,1],
		[6.922596716,1.77106367,1],
		[8.675418651,-0.242068655,1],
		[7.673756466,3.508563011,1]]

	test_data=[2.7810836,2.550537003]
	# prediction = predict_classification(dataset, dataset[0], 3)
	test_val=0
	prediction = predict_classification(dataset,test_data, 3,test_val)

	# print('Expected %d, Got %d.' % (dataset[0][-1], prediction))
	print("Usability ", prediction)

if __name__=="__main__":
	# main()
	pass

