# Clustering-Techniques-Correlation-Permutation-Signature-Graphs-Bitmaps

__Task 1:__<br>

  <i>Calculation Correlation Matrix:</i><br>
  1. Create a correlation matrix from the data matrix using Pearson’s correlation coefficient<br>
  2. The correlation matrix will be a NXN matrix (where N is number of records in your input dataset) containing Pearson’s correlation coefficient between each of the row in data matrix<br>
  3. Pearson’s correlation coefficient formula.<br><br>

  <i>Discretize:</i><br>
  1. Calculate median/mean of each column of the correlation matrix and set all the values in that column that are above the calculated median/mean to 1 and rest 0.<br><br>

  <i>Visualize:</i><br> 
  1. Convert the discretized matrix into bitmap. Sample image follow.<br>
  2. Provide functionality for zooming.<br>
  3. Display the color coded image of similarly matrix. Follow the following steps to display color coded image.<br>
        o For each column in matrix (adjacency matrix of graph), find max value.<br>
        o Divide each value in column by max value and multiply it with 255.<br>
        o Resulting values will be in range 0 to 255.<br>
        o Use this value for applying green shade to pixel.<br>
        o Sample image follow.<br><br>
    
__Task 2:__<br>

  <i>Permutate:</i><br>
  1. Permute the Data Matrix<br>
        o Do this by shuffling the individual rows in the dataset.<br>
  2. Display color coded image of permuted Data Matrix<br><br>

  <i>Apply Signature Technique:</i><br>
  1. Recover the image clusters using Signature technique. The method to generate the signature is as under.<br>
        o Sum all the values in a row<br>
        o Calculate mean of the row<br>
        o Multiply the Sum of the row with its Mean<br>
        o The above three step produces a signature for a row<br>
  2. Rearrange (sort) the Similarity Matrix by signature value of each row.<br>
  3. Apply Task1 on the rearranged matrix<br>
  4. Display the color coded image<br>

__Task 3:__<br>

  <i>Weighted Graph:</i><br>
  1. Create a weighted graph for the permuted data set.<br>
        o Calculate correlation matrix and consider it as a graph saved in a 2D array.<br>
        o Remove the edges having weights below certain threshold, provide input option.<br>
        o Create a weighted graph where each node has a certain weight. The weight of the node (in this case) is the sum of weights of all the edges connected to it.<br>
        o After that you find the node with the highest weight and get its neighbor and this becomes your one cluster.<br>
        o Then again find weights for each node and calculate the node with the highest weight.
        o The process is repeated until we are left with no clusters.

  <i>Visualize:</i><br>
  1. Visualize each of the extracted cluster.      


