import numpy as np
import matplotlib.pyplot as plt
from numpy import ndarray
import pandas as pd
import math
import networkx as nx
import os
import shutil


def read_data(file_name: str, seperator=None) -> tuple:
    """
    File data must be of format
    line1: no. of rows
    line2: no. of cols
    line3: empty line
    subsequent lines: data
    """

    # opening .txt file
    file_handler = open(file_name, 'r')

    # reading number of rows
    rows = int(file_handler.readline())

    # reading number of columns
    cols = int(file_handler.readline())

    # skip empty line
    file_handler.readline()

    # read all lines containing the data
    data_lines = file_handler.readlines()

    # retrieve all the data from the lines and store in an array
    data_array = []
    for line in data_lines:
        data_array = data_array + line.split(sep=seperator)

    # define an empty numpy array for sample data
    data = np.empty(shape=(rows, cols))

    # read data from .txt file and store in array
    index = 0
    for i in range(rows):
        for j in range(cols):
            data[i][j] = float(data_array[index])
            index += 1

    return rows, cols, data



def write_data(file_name: str, arrayType: str, data: list) -> ndarray:
    """
    write in fomat:
    line1: no. of rows
    line2: no. of cols
    line3: empty line
    subsequent lines: data
    """

    # opening .txt file in write mode
    file_handler = open(file_name, 'w')

    # writing number of rows and columns
    rows = data.shape[0]
    columns = data.shape[1]

    # writing number of rows
    file_handler.write(str(rows))
    file_handler.write("\n")

    # writing number of columns
    file_handler.write(str(columns))
    file_handler.write("\n")
    file_handler.write("\n")

    # if Matrix is of type float then write in decimal points
    if (arrayType == "float"):
        for i in range(rows):
            for j in range(columns):
                file_handler.write(str(float(data[i][j]))+" ")
            file_handler.write("\n")

    # if Matrix is of type int
    else:
        for i in range(rows):
            for j in range(columns):
                file_handler.write(str(int(data[i][j]))+" ")
            file_handler.write("\n")



def calculate_correlation_matrix(rows: int, cols: int, data: list) -> ndarray:
    """
    Computes and returns a correlation matrix of the data.
    """

    # taking transpose because we need correlation in NxN where N is number of rows.
    theData = pd.DataFrame(data.transpose())

    # Built in function corr takes the Pearson’s correlation coefficient for each element.
    correlationMatrix = theData.corr()

    return correlationMatrix



def correlationCoefficient(dataX: list, dataY: list) -> float:
    """
    Computes and returns a correlation coefficient between two rows.
    """

    sum_X = 0
    sum_Y = 0
    sum_XY = 0
    squareSum_X = 0
    squareSum_Y = 0
    totalColumns = len(dataX)

    for i in range(totalColumns):
        sum_X = sum_X + dataX[i]
        sum_Y = sum_Y + dataY[i]
        sum_XY = sum_XY + dataX[i] * dataY[i]

        # sum of square of array elements.
        squareSum_X = squareSum_X + dataX[i] * dataX[i]
        squareSum_Y = squareSum_Y + dataY[i] * dataY[i]

    # Formula for calculating correlation coefficient.
    numerator = (float)(totalColumns * sum_XY - sum_X * sum_Y)
    denominator = (float)(math.sqrt((totalColumns * squareSum_X -
                                     sum_X * sum_X) * (totalColumns * squareSum_Y - sum_Y * sum_Y)))
    correlate_coefficient = (numerator/denominator)

    return correlate_coefficient



def  pearsonCorrelationCoefficientMatrix(rows: int, data: ndarray) -> ndarray:
    """
    Computes and returns a correlation matrix between N rows.
    """

    correlationMatrix = np.empty(shape=(rows, rows))

    # compute correlationCoefficient for each row in the data.
    for i in range(rows):
        for j in range(rows):
            correlationMatrix[i][j] = correlationCoefficient(
                dataX=data[i], dataY=data[j])

    return correlationMatrix



def discretize(totalRows: int, corr_matrix: list) -> ndarray:
    """
    Computes and returns a discretize matrix of the data.
    If the value is >= mean of the col, it's placed as 0 else , 1.
    """

    discretized = np.empty(shape=(totalRows, totalRows))

    # compute descritizedOutput for each value in a column based in its mean
    for i in range(totalRows):
        sum = 0
        mean = 0

        # computing sum of each column
        for j in range(totalRows):
            sum += corr_matrix[j][i]

        mean = sum/totalRows

        # doing the opposite cuz it matches the output ƪ(˘⌣˘)ʃ
        for k in range(totalRows):
            # Going to every row of a specific column. Take a paper pencil to visualize.
            if corr_matrix[k][i] >= mean:
                discretized[k][i] = 0
            else:
                discretized[k][i] = 1

    return discretized



def colourCodedImage(totalRows: int, corr_matrix: list) -> ndarray:
    """
    Computes and returns a pixel matrix of the data.
    Divide each element by max of that col and multiply with 255.
    """

    pixelMatrix = np.empty(shape=(totalRows, totalRows, 3))

    # compute pixel value for each value in a column based in its mean
    for i in range(totalRows):
        max = 0
        # Compute the max element of each row.
        for j in range(totalRows):
            if corr_matrix[j][i] > max:
                max = corr_matrix[j][i]

        # Compute pixel value and place it in green part domain of RGB.
        for k in range(totalRows):
            # pixelMatrix[k][i]= [R                G               B]
            pixelMatrix[k][i] = [0, ((corr_matrix[k][i]/max)*255), 0]

    return pixelMatrix



def permutateMatrix(data: list) -> list:
    """
    This provide constant shuffle over a period of time. Shuffle the rows of data.
    A random seed is used to ensure that results are reproducible. In other words, 
    using this parameter makes sure that anyone who re-runs your code will get the 
    exact same outputs.
    """
    np.random.seed(50)
    np.random.shuffle(data)

    return data



def signatureCalculation(number_of_rows: int, number_of_cols: int, shuffledData: list) -> list:
    """
    Computes and returns shuffled data based on it's signature value.
    """

    signatures = np.empty(shape=(number_of_rows))

    for i in range(number_of_rows):
        rowSum = 0
        rowMean = 0
        for j in range(number_of_cols):
            rowSum += shuffledData[i][j]

        # Formula to calculate signature value of the row
        rowMean = rowSum / number_of_cols
        signatures[i] = rowSum * rowMean

    # Formula to calculate signature value of the row
    for i in range(number_of_rows-1):
        for j in range(i+1):

            if (signatures[i] > signatures[j]):
                # Swap the rows of the shuffledData, swap each value in the column.
                for k in range(number_of_cols):
                    shuffledData[i][k], shuffledData[j][k] = shuffledData[j][k], shuffledData[i][k]

                # Swap their respective signature values.
                signatures[i], signatures[j] = signatures[j], signatures[i]

    return shuffledData



def setPermutatedMatrix(number_of_rows: int, number_of_cols: int, shuffledData: list) -> list:
    """
    Computes and returns weighted matrix based on threshold value.
    """

    threshHold = float(input("Enter the threshold: "))

    weightedMatrix = np.empty(shape=(number_of_rows, number_of_rows))

    for i in range(number_of_rows):
        for j in range(number_of_cols):

            # Make the value below threshold 0.
            if shuffledData[i][j] < threshHold:
                weightedMatrix[i][j] = 0
            else:
                weightedMatrix[i][j] = shuffledData[i][j]

    return weightedMatrix



def getNodeWeights(number_of_rows: int, number_of_cols: int, weightedMatrix: list) -> list:
    """
    Computes and returns node weights.
    I was thinking to -1 from each nodeWeight,
    until i realize 1 is present in each row,
    p.s: correlationMatrix from the row to itself is 1 ✍️(◔◡◔)
    """

    nodeWeights = np.empty(shape=(number_of_rows))

    for i in range(number_of_rows):
        for j in range(number_of_cols):
            nodeWeights[i] += weightedMatrix[i][j]

    return nodeWeights



def getIndexOfMaxWeight(number_of_rows: int, nodeWeights: list) -> int:

    """
    Computes and returns maximum index of the max value in NodeWeights.
    """

    max = -50
    index = -1

    for i in range(number_of_rows):
        if nodeWeights[i] > max:
            max = nodeWeights[i]
            index = i

    # print(index+1)
    # setting -1 to that index so that it doesn't comes again.
    nodeWeights[index] = -1

    return index



def makeClusters(index: int, graphList: list):

    """
    Makes the Graph of the graphList and visualize it as well.
    """

    # Initialize directed weighted graph.
    graph = nx.DiGraph()
    sizeOfGraphList = len(graphList)

    #Add the index with max nodeWeight to graph first. 
    graph.add_node(str(index+1))

    for i in range(sizeOfGraphList):

        # If no path exists or graphList = 1 indicates the graph of that index with itself
        if graphList[i] == float(0) or graphList[i] == float(1):
            # Go ↑.
            continue                                

        else:
            # Add the node connected to source.
            graph.add_node(str(i+1))
            # Add edge between them representing the weight between them.
            graph.add_edge(str(index+1), str(i+1), weight=graphList[i])

    # print(list(graph.nodes))
    # print(list(graph.edges))

    # index+1 makes 0-149, 1-150. Just seems good in visualization.
    visualizeWeightedGraph(G=graph, index=index+1)



def visualizeWeightedGraph(G, index: int):
    """
    Visualize the graph and save it in png form.
    """

    plt.figure()
    position = nx.spring_layout(G)

    # Draws a directed graph from source to destination.
    nx.draw(G, position, font_color='white', node_shape='s', with_labels=True,)

    # Draws edge weights from source to destination.
    # weight_labels = nx.get_edge_attributes(G, 'weight')
    # output = nx.draw_networkx_edge_labels(
    #     G, position, edge_labels=weight_labels)

    # Save the png of the cluster in the path below.
    plt.savefig(
        "./Task-3-Output/GraphClusterVisualization/Cluster-"+str(index)+".png")
    
    # Closing just because it throws error on opening more than 20 file.
    plt.close()
    # For visualization here, uncomment the line below.
    # plt.show()



def runTaskOne(fileName: str):
    """
    Runs all the requirements of Task-1 as in Question Statement.
    """

    # Reading data.
    number_of_rows, number_of_columns, sample_data = read_data(
        file_name=fileName,
    )

    # Initialization of 2-D Array.
    correlationMatrix = np.empty(shape=(number_of_rows, number_of_rows))

    # CalculatingCorrelationMatrix.
    correlationMatrix =  pearsonCorrelationCoefficientMatrix(
        rows = number_of_rows, data=sample_data)

    # Writing output to text file.
    write_data("./Task-1-Output/CorrelationMatrix.txt",
               "float", correlationMatrix)


    # Initialization of 2-D Array.
    discretizedMatrix = np.empty(shape=(number_of_rows, number_of_rows))

    # CalculatingDiscritizedMatrix.
    discretizedMatrix = discretize(number_of_rows, correlationMatrix)

    # Writing output to text file.
    write_data("./Task-1-Output/DescritizedMatrix.txt",
               "int", discretizedMatrix)


    # Initialization of 3-D Array.
    pixelMatrix = np.empty(shape=(number_of_rows, number_of_rows, 3))

    # CalculatingPixelMatrix.
    pixelMatrix = colourCodedImage(number_of_rows, correlationMatrix)

    # Plotting Discritized and Color Matrix.
    fig, axes = plt.subplots(nrows=1, ncols=2)
    axes[0].set_title('Discretized Matrix')
    axes[0].imshow(discretizedMatrix, cmap='gray')
    axes[1].set_title('Color Coded Correlation Matrix')
    axes[1].imshow(pixelMatrix.astype(int))
    plt.show()



def runTaskTwo(fileName: str):
    """
    Runs all the requirements of Task-2 as in Question Statement.
    """

    # Reading data.
    number_of_rows, number_of_columns, temp_data = read_data(
        file_name=fileName,
    )
    # Reading data.
    number_of_rows, number_of_columns, sample_data = read_data(
        file_name=fileName,
    )

    # Permutating/Shuffling the rows in input data randomly.
    shuffledData = permutateMatrix(temp_data)

    # Writing output to text file.
    write_data("./Task-2-Output/InputAfterShuffling.txt",
               "float", shuffledData)


    # Initialization of 2-D Array.
    correlationMatrixBeforePermutation = np.empty(
        shape=(number_of_rows, number_of_rows))

    # CalculatingCorrelationMatrix Before Permutation.
    correlationMatrixBeforePermutation =  pearsonCorrelationCoefficientMatrix(
        rows = number_of_rows, data=sample_data)

    # Initialization of 2-D Array.
    correlationMatrixAfterPermutation = np.empty(
        shape=(number_of_rows, number_of_rows))

    # CalculatingCorrelationMatrix After Permutation.
    correlationMatrixAfterPermutation =  pearsonCorrelationCoefficientMatrix(
        rows = number_of_rows, data=shuffledData)

    # Writing output to text file.
    write_data("./Task-2-Output/CorrelationMatrixAfterPermutation.txt",
               "float", correlationMatrixAfterPermutation)

    
    # Initialization of 2-D Array.
    discretizedMatrixBeforePermutation = np.empty(
        shape=(number_of_rows, number_of_rows))

    # CalculatingDiscritizedMatrix Before Permutation.
    discretizedMatrixBeforePermutation = discretize(
        number_of_rows, correlationMatrixBeforePermutation)

    
    # Initialization of 2-D Array.
    discretizedMatrixAfterPermutation = np.empty(
        shape=(number_of_rows, number_of_rows))

    # CalculatingDiscritizedMatrix After Permutation.
    discretizedMatrixAfterPermutation = discretize(
        number_of_rows, correlationMatrixAfterPermutation)

    # Writing output to text file.
    write_data("./Task-2-Output/DescritizedMatrixAfterPermutation.txt",
               "int", discretizedMatrixAfterPermutation)

    
    # Plotting Discritized Matrix before and after permutation.
    fig, axes = plt.subplots(nrows=1, ncols=2)
    axes[0].set_title('Before Permutation')
    axes[0].imshow(discretizedMatrixBeforePermutation, cmap='gray')
    axes[1].set_title('After Permutation')
    axes[1].imshow(discretizedMatrixAfterPermutation, cmap='gray')
    plt.show()

   
    # Calculating and Sorting input data after permutation 
    # with respect to the signature value of respective row.
    sortedViaSignatured = signatureCalculation(number_of_rows=number_of_rows,
                                               number_of_cols=number_of_columns, shuffledData=shuffledData)
    # Writing output to text file.
    np.savetxt("./Task-2-Output/InputSortedViaSignature.txt",
               sortedViaSignatured, fmt='%.1f')

    
    # Initialization of 2-D Array.
    correlationMatrix = np.empty(
        shape=(number_of_rows, number_of_rows))

    # CalculatingCorrelationMatrix After Permutation.
    correlationMatrix =  pearsonCorrelationCoefficientMatrix(
        rows = number_of_rows, data=sortedViaSignatured)

    # Writing output to text file.
    write_data("./Task-2-Output/SignatureCorrelationMatrix.txt",
               "float", correlationMatrix)

    
    # Initialization of 3-D Array.
    pixelMatrix = np.empty(shape=(number_of_rows, number_of_rows, 3))

    # CalculatingPixelMatrix.
    pixelMatrix = colourCodedImage(number_of_rows, correlationMatrix)

    # Plotting Pixel Matrix after SignatureWiseSorting And Rearrangement. 
    plt.suptitle('After Signature-Wise-Sorting')
    plt.imshow(pixelMatrix.astype(int))
    plt.show()



def runTaskThree(fileName: str):
    """
    Runs all the requirements of Task-3 as in Question Statement.
    """

    # Reading data.
    number_of_rows, number_of_columns, temp_data = read_data(
        file_name=fileName,
    )

    # Permutating/Shuffling the rows in input data randomly.
    shuffledData = permutateMatrix(temp_data)

    # Writing output to text file.
    write_data("./Task-3-Output/InputAfterShuffling.txt",
               "float", shuffledData)


    # Initialization of 2-D Array.
    correlationMatrix = np.empty(
        shape=(number_of_rows, number_of_rows))

    # CalculatingCorrelationMatrix.
    correlationMatrix =  pearsonCorrelationCoefficientMatrix(
        rows = number_of_rows, data=shuffledData)

    # Writing output to text file.
    write_data("./Task-3-Output/CorrelationMatrix.txt",
               "float", correlationMatrix)


    # Initialization of 2-D Array.
    updatedCorrelationMatrix = np.empty(
        shape=(number_of_rows, number_of_rows))

    # CalculatingUpdatedCorrelationMatrix after setting it w.r.to threshold.
    updatedCorrelationMatrix = setPermutatedMatrix(
        number_of_rows=number_of_rows, number_of_cols=number_of_rows, shuffledData=correlationMatrix)

    # Writing output to text file.
    write_data("./Task-3-Output/ThresholdCorrelationMatrix.txt",
               "float", updatedCorrelationMatrix)


    # Initialization of 2-D Array.
    nodeWeights = np.empty(shape=(number_of_rows))

    # CalculatingNodeWeights of updatedCorrelationMatrix.
    nodeWeights = getNodeWeights(
        number_of_rows=number_of_rows, number_of_cols=number_of_rows, weightedMatrix=updatedCorrelationMatrix)

    # Writing output to text file.
    np.savetxt("./Task-3-Output/NodeWeights.txt", nodeWeights, fmt='%f')

    '''
    Now i have two options here,
    1- I can either sort the correlationMatrix and NodeWeights
    2- Or i can find the index with highest nodeWeight and then make it's graph according to its index.
    Better is the option 2, bcz sorting would increase the TimeComplexity
    I'd have to swap complete N*N matrix, 150*2(CorrelationMatrix) + 2(NodeWeights) in each row,
    Thus, option 1 is a better approach.
    '''

    # removing if the directory is already present
    shutil.rmtree('./Task-3-Output/GraphClusterVisualization',
                  ignore_errors=True)

    # creating a directory for saving png file of clusters
    os.mkdir('./Task-3-Output/GraphClusterVisualization')

    # GettingIndex Of the row with maximum weight and then making cluster of it.
    # After that again search for index of row with maximum weight and make the
    # cluster of it. And so on until we loop through each row. 
    for i in range(number_of_rows):
        index = getIndexOfMaxWeight(
            number_of_rows=number_of_rows, nodeWeights=nodeWeights)

        if(i == 0):
            print(f"\nMost densed cluster: {index+1}\n")

        if(i == 149):
            print(f"\nLeast densed cluster: {index+1}\n")

        makeClusters(index=index, graphList=updatedCorrelationMatrix[index])


# Using the while loop to print menu list

def main():

    # Default Input File
    fileName = 'Sample data-1-IRIS.TXT'

    while True:
        print("\n-------- Menu --------")
        print("\n- Select Input File -")
        print("1. Sample data-1-IRIS")
        print("2. Sample data-2-INPUT1")
        print("3. Sample data-3-VINE")

        choice = int(input("\nEnter your choice(1-3): "))

        if choice == 1:
            fileName = 'Sample data-1-IRIS.TXT'
            break

        elif choice == 2:
            fileName = 'Sample data-2-INPUT1.TXT'
            break

        elif choice == 3:
            fileName = 'Sample data-3-VINE.TXT'
            break

        else:
            print("Incorrect Choice. Please, try again.\n")


    while True:
        print("\n\n- SubMenu -\n")
        print("1. Task-01")
        print("2. Task-02")
        print("3. Task-03")
        print("4. Exit")

        choice = int(input("\nEnter your choice(1-4): "))

        if choice == 1:
            runTaskOne(fileName=fileName)

        elif choice == 2:
            runTaskTwo(fileName=fileName)

        elif choice == 3:
            runTaskThree(fileName=fileName)

        elif choice == 4:
            print("\nThank You! See you again.\n")
            break

        else:
            print("Incorrect Choice. Please, try again.")

if __name__ == "__main__":
         main()
