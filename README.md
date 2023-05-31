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
    3. Recover the image clusters using Signature technique. The method to generate the signature is as under.<br>
              o Sum all the values in a row<br>
              o Calculate mean of the row<br>
              o Multiply the Sum of the row with its Mean<br>
              o The above three step produces a signature for a row<br>
    4. Rearrange (sort) the Similarity Matrix by signature value of each row.<br>
    5. Apply Task1 on the rearranged matrix<br>
    6. Display the color coded image<br>
      
