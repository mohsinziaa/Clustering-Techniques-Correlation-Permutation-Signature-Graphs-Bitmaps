# Clustering-Techniques-Correlation-Permutation-Signature-Graphs-Bitmaps

Task 1:

  Calculation Correlation Matrix:
    1. Create a correlation matrix from the data matrix using Pearson’s correlation coefficient
    2. The correlation matrix will be a NXN matrix (where N is number of records in your input dataset) containing Pearson’s correlation coefficient between each of          the row in data matrix
    3. Pearson’s correlation coefficient formula.  

  Discretize:
    1. Calculate median/mean of each column of the correlation matrix and set all the values in that column that are above the calculated median/mean to 1 and rest 0.

  Visualize: 
    1. Convert the discretized matrix into bitmap. Sample image follow.
    2. Provide functionality for zooming.
    3. Display the color coded image of similarly matrix. Follow the following steps to display color coded image.
        o For each column in matrix (adjacency matrix of graph), find max value.
        o Divide each value in column by max value and multiply it with 255.
        o Resulting values will be in range 0 to 255.
        o Use this value for applying green shade to pixel.
        o Sample image follow.
      
Task 2:

    1. Permute the Data Matrix
        o Do this by shuffling the individual rows in the dataset.
    2. Display color coded image of permuted Data Matrix
    3. Recover the image clusters using Signature technique. The method to generate the signature is as under.
        o Sum all the values in a row
        o Calculate mean of the row
        o Multiply the Sum of the row with its Mean
        o The above three step produces a signature for a row
    4. Rearrange (sort) the Similarity Matrix by signature value of each row.
    5. Apply Task1 on the rearranged matrix
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    ![image](https://user-images.githubusercontent.com/75737591/232343799-4870779c-2817-4c02-ac24-8ebfbdaedcf7.png)

    6. Display the color coded image
      
