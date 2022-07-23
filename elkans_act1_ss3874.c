// Elkan's algorithm Parallelized


#include <stdio.h>
#include <unistd.h>
#include <mpi.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>

#define KMEANSITERS 10

// compile
// mpicc elkans_act1_ss3874.c -lm -o elkans_act1_ss3874

// run example with 10 means
// mpirun -np 4 -hostfile myhostfile.txt ./elkans_act1_ss3874 5159737 2 10 iono_57min_5.16Mpts_2D.txt

// function prototypes
int importDataset(char *fname, int DIM, int N, double **dataset);

int main(int argc, char **argv)
{

    int my_rank, nprocs;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &nprocs);

    // Process command-line arguments
    int N;
    int DIM;
    int KMEANS;
    char inputFname[500];

    if (argc != 5)
    {
        fprintf(stderr, "Please provide the following on the command line: N (number of lines in the file), dimensionality (number of coordinates per point/feature vector), K (number of means), dataset filename. Your input: %s\n", argv[0]);
        MPI_Finalize();
        exit(0);
    }

    sscanf(argv[1], "%d", &N);
    sscanf(argv[2], "%d", &DIM);
    sscanf(argv[3], "%d", &KMEANS);
    strcpy(inputFname, argv[4]);

    // pointer to entire dataset
    double **dataset;

    if (N < 1 || DIM < 1 || KMEANS < 1)
    {
        printf("\nOne of the following are invalid: N, DIM, K(MEANS)\n");
        MPI_Finalize();
        exit(0);
    }
    // All ranks import dataset
    else
    {

        if (my_rank == 0)
        {
            printf("\nNumber of lines (N): %d, Dimensionality: %d, KMEANS: %d, Filename: %s\n", N, DIM, KMEANS, inputFname);
        }

        // allocate memory for dataset
        dataset = (double **)malloc(sizeof(double *) * N);
        for (int i = 0; i < N; i++)
        {
            dataset[i] = (double *)malloc(sizeof(double) * DIM);
        }

        int ret = importDataset(inputFname, DIM, N, dataset);

        if (ret == 1)
        {
            MPI_Finalize();
            return 0;
        }
    }

    MPI_Barrier(MPI_COMM_WORLD);

    // Write code here

    // Variable declarations required for Kmeans Algorithm:  

    int i, j, k, l;
    int numPointsForMyRank;

    // Local Time measurement variables
    double tstart_distance_1, tend_distance_1, tstart_distance_2, tend_distance_2;
    double tstart_distance_3, tend_distance_3, tstart_distance_4, tend_distance_4;
    double tstart_distance_5, tend_distance_5, tstart_distance_6, tend_distance_6;
    double local_time_taken_distance = 0;
    double local_time_taken_updating = 0;
    double tstart_validate, tend_validate, local_time_taken_validate = 0;
    double tstart_total, tend_total, local_time_taken_total = 0;

    // Global Time measurement variables
    double global_time_taken_distance;
    double global_time_taken_updating;
    double global_time_taken_total;
    
    // To store the KMEANS number of centroids
    double **centroids;
    centroids = (double **)malloc(sizeof(double *) * (KMEANS));
    for (i = 0; i < KMEANS; i++)
    {
        centroids[i] = (double *)malloc(sizeof(double) * DIM);
    }

    int *numPointsForEachRank;

    if (my_rank == 0)
    {
        numPointsForEachRank = (int *)malloc(sizeof(int) * (nprocs));
    }

    // To find number of points given to each rank:
    int numPointsForRegularRank = N / nprocs;
    int numPointsLeftOver = N % nprocs;

    // Calculate number of points given to each rank at Rank 0:
    if (my_rank == 0)
    {
        for (i = 0; i < nprocs; i++)
        {
            numPointsForEachRank[i] = numPointsForRegularRank;

            // Distribute leftover points over other ranks
            // Ex.: If 5 points leftover, give one extra point to first 5 ranks.
            if (i < numPointsLeftOver)
            {
                numPointsForEachRank[i] = numPointsForEachRank[i] + 1;
            }
        }
    }

    // Send number of points for each rank from rank 0
    MPI_Scatter(numPointsForEachRank, 1, MPI_INT, &numPointsForMyRank, 1, MPI_INT, 0, MPI_COMM_WORLD);

    int myStartPoint;

    if (my_rank < numPointsLeftOver)
    {
        myStartPoint = numPointsForMyRank * my_rank;
    }
    else
    {
        myStartPoint = (numPointsForRegularRank * my_rank) + numPointsLeftOver;
    }

    // To store the assigned centroid for each point
    // Store the Array Index of the assigned Centroid instead of values for all dimensions.
    int *assignedCentroid;
    assignedCentroid = (int *)malloc(sizeof(int *) * numPointsForMyRank);

    double distance, minimumDistance_xc;

    // To store the sum of coordinates of the points assigned to a Centroid at each rank
    double **localSumOfCentroidPoints;
    localSumOfCentroidPoints = (double **)malloc(sizeof(double *) * KMEANS);
    for (i = 0; i < KMEANS; i++)
    {
        localSumOfCentroidPoints[i] = (double *)malloc(sizeof(double) * DIM);
    }

    // To store the number of points assigned to a Centroid at each rank
    int *localNumOfPointsForCentroid;
    localNumOfPointsForCentroid = (int *)malloc(sizeof(int) * KMEANS);

    // To store the Global Sum of coordinates of the points assigned to a Centroid 
    double **globalSumOfCentroidPoints;
    globalSumOfCentroidPoints = (double **)malloc(sizeof(double *) * KMEANS);
    for (i = 0; i < KMEANS; i++)
    {
        globalSumOfCentroidPoints[i] = (double *)malloc(sizeof(double) * DIM);
    }

    // To store the number of points assigned to a Centroid globally
    int *globalNumOfPointsForCentroid;
    globalNumOfPointsForCentroid = (int *)malloc(sizeof(int) * KMEANS);


    // Additional Variable declarations for Elkan's Algorithm:  

    // To store the distance between the centers at each iteration
    double **distance_c1c2;
    distance_c1c2 = (double **)malloc(sizeof(double *) * (KMEANS));
    for (i = 0; i < KMEANS; i++)
    {
        distance_c1c2[i] = (double *)malloc(sizeof(double) * KMEANS);
    }

    // To store the distance between a point and its assigned centroid c(x) 
    // or c1, d(x,c(x)) 
    double distance_xc1;

    // To store the distance between a point and its new centroid
    // which may or may not be assigned to x, i.e. d(x,c) 
    double distance_xc;

    // To store the new values of KMEANS number of centroids, c'
    double **centroidsNew;
    centroidsNew = (double **)malloc(sizeof(double *) * (KMEANS));
    for (i = 0; i < KMEANS; i++)
    {
        centroidsNew[i] = (double *)malloc(sizeof(double) * DIM);
    }

    // To store the distance d(c,c') between old value of centroid and 
    // new value of centroid calculated as mean of points assigned to it 
    double *distance_ccdash;
    distance_ccdash = (double *)malloc(sizeof(double) * (KMEANS));

    // Lower bound l(x,c) for each point x and center c
    double **lower_bound;
    lower_bound = (double **)malloc(sizeof(double *) * (numPointsForMyRank));
    for (i = 0; i < numPointsForMyRank; i++)
    {
        lower_bound[i] = (double *)malloc(sizeof(double) * KMEANS);
    }

    // Upper bound u(x) for each point x
    double *upper_bound;
    upper_bound = (double *)malloc(sizeof(double) * (numPointsForMyRank));

    // s(c) given by (1/2) of min value of d(c1, c2)
    double *s_of_centroid;
    s_of_centroid = (double *)malloc(sizeof(double) * (KMEANS));

    double minimumDistance_c1c2;
    
    // r(x), used as a flag to check if distance between point and 
    // assigned centroid, i.e. d(x,c(x)) is outdated or not.
    int *r_of_x;
    r_of_x = (int *)malloc(sizeof(int) * (numPointsForMyRank));
    for (i = 0; i < numPointsForMyRank; i++)
    {
        r_of_x[i] = 1;
    }

    double **swap;

    // Flag to check if convergence has been reached or not
    int convergence_reached = 0;

    // For validation purposes:
    double maxCentroid, minCentroid;
    int flag = 0;
    int distancecalculations = 0, globaldistancecalculations = 0;

    // Start measuring time on all ranks
    tstart_total = MPI_Wtime();


    // Initialization of Elkan's algorithm

    // Pick initial centres:
    // Assign initial values for the k centroids as the first KMEANS points
    // of the dataset
    for (i = 0; i < KMEANS; i++)
    {
        for (j = 0; j < DIM; j++)
        {
            centroids[i][j] = dataset[i][j];
        }
    }

    // Print the initial values of centroids:
    if (my_rank == 0)
    {
        printf("\nInitial Values of Centroids");
        for (i = 0; i < KMEANS; i++)
        {
            printf("\nC%d\t", i);
            for (j = 0; j < DIM; j++)
            {
                printf("%f\t", centroids[i][j]);
            }
        }
        printf("\n");
    }

    // Distance c1c2 matrix calculation 
    // To use in Centroid assignment step and reduce number of d(x,c) calculations

    tstart_distance_1 = MPI_Wtime();

    for (j = 0; j < KMEANS; j++)
    {
        for (k = 0; k < KMEANS; k++)
        {
            double dimsquares_cc = 0;
            for (l = 0; l < DIM; l++)
            {
                dimsquares_cc = dimsquares_cc + pow((centroids[j][l] - centroids[k][l]), 2);
            }
            distance_c1c2[j][k] = sqrt(dimsquares_cc);
            distancecalculations = distancecalculations + 1;
        }
    }

    tend_distance_1 = MPI_Wtime();
    local_time_taken_distance = local_time_taken_distance + tend_distance_1 - tstart_distance_1;
    

    for (j = 0; j < numPointsForMyRank; j++)
    {
        for (k = 0; k < KMEANS; k++)
        {
            // Assign lower bound l(x,c) = 0 for all x and c.
            lower_bound[j][k] = 0;

            // Assign each point to its closest centroid
            // Use Lemma 1 to reduce number of distance calculations
            // i.e. Do not calculate d(x,c2) if d(x,c1) <= (1/2)* d(c1,c2)
            if ((k == 0)||(minimumDistance_xc > ((1/2) * distance_c1c2[assignedCentroid[j]][k])))
            {            
                tstart_distance_2 = MPI_Wtime();

                // Calculate distance of each point to centroid if needed
                double dimsquares = 0.0;
                for (l = 0; l < DIM; l++)
                {
                    dimsquares = dimsquares + pow((dataset[myStartPoint + j][l] - centroids[k][l]), 2);
                }
                distance = sqrt(dimsquares);
                
                // Assign l(x,c) if distance is calculated
                lower_bound[j][k] = distance;

                tend_distance_2 = MPI_Wtime();
                local_time_taken_distance = local_time_taken_distance + tend_distance_2 - tstart_distance_2;

                distancecalculations = distancecalculations + 1;

                // Assign x to closest initial center c(x)
                if (k == 0)
                {
                    assignedCentroid[j] = k;
                    minimumDistance_xc = distance;
                }
                else
                {
                    if (distance < minimumDistance_xc)
                    {
                        assignedCentroid[j] = k;
                        minimumDistance_xc = distance;
                    }
                }
            }
        } 

        // Assign upper bound of u(x) = min d(x,c)
        upper_bound[j] = minimumDistance_xc;    
    }

    // Initialization Complete


    // Iterations of Elkan - Repeat until convergence

    i = 0;

    
    // while (!convergence_reached)
    for (i = 0; i < KMEANSITERS; i++)
    {
        // Initialize (or) reinitialize the values of Sum variables to zero.

        for (j = 0; j < KMEANS; j++)
        {
            for (k = 0; k < DIM; k++)
            {
                localSumOfCentroidPoints[j][k] = 0;
                globalSumOfCentroidPoints[j][k] = 0;
            }
            localNumOfPointsForCentroid[j] = 0;
            globalNumOfPointsForCentroid[j] = 0;
        }

        // Step 1(i): For all centres c1 and c2, compute distances d(c1,c2).

        for (j = 0; j < KMEANS; j++)
        {
            for (k = 0; k < KMEANS; k++)
            {
                tstart_distance_3 = MPI_Wtime();

                double dimsquares_cc = 0;
                for (l = 0; l < DIM; l++)
                {
                    dimsquares_cc = dimsquares_cc + pow((centroids[j][l] - centroids[k][l]), 2);
                }
                distance_c1c2[j][k] = sqrt(dimsquares_cc);

                tend_distance_3 = MPI_Wtime();
                local_time_taken_distance = local_time_taken_distance + tend_distance_3 - tstart_distance_3;

                distancecalculations = distancecalculations + 1;

                // (ii) For finding s(c), find min d(c1,c2)   
                if (j != k)
                {
                    if((k == 0)||((j == 0)&&(k == 1)))
                    {
                        minimumDistance_c1c2 = distance_c1c2[j][k];
                    }
                    else
                    {
                        if (distance_c1c2[j][k] < minimumDistance_c1c2)
                        {
                            minimumDistance_c1c2 = distance_c1c2[j][k];
                        }
                    }
                }
            }
            s_of_centroid[j] = (1/2) * minimumDistance_c1c2;
        }

        //  Step 2: Identify points x such that u(x) <= s(c(x))
        // Further calculation is not needed for such points.

        for (j = 0; j < KMEANS; j++)
        {
            for (k = 0; k < numPointsForMyRank; k++)
            {
                if (!(upper_bound[k] <= s_of_centroid[assignedCentroid[k]]))                                    
                {

                    //  Step 3: If point x satisfies given 3 conditions, then continue:
                    // Condition 1: c != c(x)
                   if (j != assignedCentroid[k])
                   {
                        // Condition 2: u(x) > l(x,c)
                       if (upper_bound[k] > lower_bound[k][j])
                       {
                            // Condition 3: u(x) > (1/2) d(c(x),c)
                           if (upper_bound[k] > ((1/2) * distance_c1c2[assignedCentroid[k]][j]))
                           {
                               // Step 3a: If all conditions satisfied and flag r(x) is true, find d(x,c(x)).
                               if (r_of_x[k] == 1)
                               {
                                    tstart_distance_4 = MPI_Wtime();

                                    double dimsquares = 0.0;
                                    for (l = 0; l < DIM; l++)
                                    {
                                        dimsquares = dimsquares + pow((dataset[myStartPoint + k][l] - centroids[assignedCentroid[k]][l]), 2);
                                    }
                                    distance_xc1 = sqrt(dimsquares);

                                    // Set upper bound each time d(x,c(x)) is calculated.
                                    upper_bound[k] = distance_xc1;
                                    r_of_x[k] = 0;

                                    tend_distance_4 = MPI_Wtime();
                                    local_time_taken_distance = local_time_taken_distance + tend_distance_4 - tstart_distance_4;

                                    distancecalculations = distancecalculations + 1;
                               }
                               else
                               {
                                //    If flag r(x) is false, use previous value stored in u(x) as d(x,c(x))
                                   distance_xc1 = upper_bound[k];
                               }

                               // Step 3b: If d(x,c(x)) > l(x,c) or (1/2) * d(c(x),c), find d(x,c)).
                               if ((distance_xc1 > lower_bound[k][j])||(distance_xc1 > ((1/2) * distance_c1c2[assignedCentroid[k]][j])))
                               {
                                    tstart_distance_5 = MPI_Wtime();

                                    double dimsquares = 0.0;
                                    for (l = 0; l < DIM; l++)
                                    {
                                        dimsquares = dimsquares + pow((dataset[myStartPoint + k][l] - centroids[j][l]), 2);
                                    }
                                    distance_xc = sqrt(dimsquares);

                                    // Set lower bound each time d(x,c) is calculated.
                                    lower_bound[k][j] = distance_xc;
                                    
                                    tend_distance_5 = MPI_Wtime();
                                    local_time_taken_distance = local_time_taken_distance + tend_distance_5 - tstart_distance_5;

                                    distancecalculations = distancecalculations + 1;
                                    
                                    // If new d(x,c) is less than distance to assigned centroid, then reassign x to new centroid.
                                    // Change flag r(x) to 1 to indicate that d(x,c(x)) is outdated.
                                    if(distance_xc < distance_xc1)
                                    {
                                        assignedCentroid[k] = j;
                                        r_of_x[k] = 1;
                                    }
                               }
                           }
                       }
                   } 
                }
            }
        }


        // Step 4: Calculate new value of centroid as weighted mean of points 
        // assigned to the centroid
        
        // Calculate sum of point coordinates and number of points assigned to each centroid
        for (k = 0; k < numPointsForMyRank; k++)
        {
            for (l = 0; l < DIM; l++)
            {
                localSumOfCentroidPoints[assignedCentroid[k]][l] = localSumOfCentroidPoints[assignedCentroid[k]][l] + dataset[myStartPoint + k][l];
            }
            localNumOfPointsForCentroid[assignedCentroid[k]] = localNumOfPointsForCentroid[assignedCentroid[k]] + 1;
        }

        // Calculate global sums of point coordinates and number of points for each centroid
        for (j = 0; j < KMEANS; j++)
        {
            for (k = 0; k < DIM; k++)
            {
                MPI_Allreduce(&(localSumOfCentroidPoints[j][k]), &(globalSumOfCentroidPoints[j][k]), 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
            }
            MPI_Allreduce(&(localNumOfPointsForCentroid[j]), &(globalNumOfPointsForCentroid[j]), 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
        }

        // Use global sums to compute new Centroid, store in centroidsNew:
        for (j = 0; j < KMEANS; j++)
        {
            for (l = 0; l < DIM; l++)
            {
                if (globalNumOfPointsForCentroid[j] != 0)
                {
                    centroidsNew[j][l] = globalSumOfCentroidPoints[j][l] / globalNumOfPointsForCentroid[j];
                }
                else
                {
                    // If no points have been assigned to the Centroid, Division by Zero Error may occur
                    // Instead of dividing by 0,re-initialize centroid to Origin.
                    centroidsNew[j][l] = 0;
                }
            }

            
            // Calculate distances between the old and new values of each centroid 
            // to use in Step 5 and Step 6:
             
            tstart_distance_6 = MPI_Wtime();

            double dimsquares = 0.0;
            for (l = 0; l < DIM; l++)
            {
                dimsquares = dimsquares + pow((centroidsNew[j][l] - centroids[j][l]), 2);
            }
            distance_ccdash[j] = sqrt(dimsquares);

            tend_distance_6 = MPI_Wtime();
            local_time_taken_distance = local_time_taken_distance + tend_distance_6 - tstart_distance_6;
            
            distancecalculations = distancecalculations + 1;

        }

        // Step 5: Calculate new lower bound l(x,c) as max {l(x,c) - d(c,m(c)), 0}
        for ( k = 0; k < numPointsForMyRank; k++)
        {
            for (j = 0; j < KMEANS; j++)
            {
                if ((lower_bound[k][j] - distance_ccdash[j]) > 0)
                {
                    lower_bound[k][j] =  lower_bound[k][j] - distance_ccdash[j];
                }
                else
                {
                    lower_bound[k][j] = 0;
                }
            }
            // Step 6: Calculate new upper bound u(x) as u(x) + d(c(x),m(c(x)))
            upper_bound[k] = upper_bound[k] + distance_ccdash[assignedCentroid[k]];
            r_of_x[k] = 1;
        }

        // Check if convergence is reached:
        // Compare old centroids with new
        convergence_reached = 1;
        for (j = 0; j < KMEANS; j++)
        {
            for (l = 0; l < DIM; l++)
            {
                if (centroids[j][l] != centroidsNew[j][l])
                    convergence_reached = 0;
            }
        }


        // Step 7: Replace each center c by m(c):

        // Swap pointer of centroids with centroidsNew
        swap = centroids; centroids = centroidsNew; centroidsNew = swap;
      
        // Validation:

        // Start measuring distance calculation time on all ranks
        tstart_validate = MPI_Wtime();

        // Print New Centroids:
        if (my_rank == 0)
        {
            printf("\nCentroids in Iteration %d:\n", i);

            for (j = 0; j < KMEANS; j++)
            {
                printf("Centroid %d : ", j);
                for (k = 0; k < DIM; k++)
                {
                    printf("\t%f", centroids[j][k]);
                }
                printf("   : Number of Points assigned to C%d = %d\n", j, globalNumOfPointsForCentroid[j]);
            }
        }

        // Check if Centroids are same on all ranks at each iteration:
        // Perform Maximum and Minimum reduction on the Centroids, if they are both the same and equal
        // to the Centroid, the centroid values are same on all ranks.

        flag = 0;

        for (j = 0; j < KMEANS; j++)
        {
            for (k = 0; k < DIM; k++)
            {
                MPI_Reduce(&(centroids[j][k]), &maxCentroid, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
                MPI_Reduce(&(centroids[j][k]), &minCentroid, 1, MPI_DOUBLE, MPI_MIN, 0, MPI_COMM_WORLD);

                if (my_rank == 0)
                {
                    if (maxCentroid != minCentroid)
                        flag = 1;
                }
            }
        }

        if (my_rank == 0)
        {
            if (flag == 0)
                printf("Values of Centroids are equal at all ranks!\n");
            else
                printf("Error: Values of Centroids are not equal at all ranks!\n");
        }

        // End of Validation - Stop measuring time for Validation
        tend_validate = MPI_Wtime();

        // Calculate time taken for Validation as (ending time - starting time)
        local_time_taken_validate = local_time_taken_validate + tend_validate - tstart_validate;

        
        if (my_rank == 0)
        {
            if (convergence_reached == 1)
                printf("\nConvergence Reached at Iteration %d!\n", i);
        }

        // Un-comment if running till convergence with while loop
        // i++;

    }

    // End of both Distance Calculation and Updating Centroids Steps - Stop measuring total time
    tend_total = MPI_Wtime();

    // Calculate total time taken as (ending time - starting time)
    local_time_taken_total = tend_total - tstart_total - local_time_taken_validate;

    // Calculate total time taken for updating centroids as (total time - distance time)
    // This is calculated this on one rank since it is difficult to separate out the 
    // distance calculations from other calculations and measure accurately.
    local_time_taken_updating = local_time_taken_total - local_time_taken_distance;


    // Use MPI_Reduce to compute maximum time taken among the ranks for Distance Calculation
    MPI_Reduce(&local_time_taken_distance, &global_time_taken_distance, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    // Use MPI_Reduce to compute maximum time taken among the ranks for updating
    MPI_Reduce(&local_time_taken_updating, &global_time_taken_updating, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    // Use MPI_Reduce to compute maximum time taken among the ranks in total
    MPI_Reduce(&local_time_taken_total, &global_time_taken_total, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    // Use MPI_Reduce to compute number of distance calculations
    MPI_Reduce(&distancecalculations, &globaldistancecalculations, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);

    // Print time taken for Distance Calculation, Updating Centroids, and Total Time taken:
    if (my_rank == 0)
    {
        printf("\nNumber of Distance Calculations = %d\n", globaldistancecalculations);
        printf("\nTime taken for Distance Calculation = %f\n", global_time_taken_distance);
        printf("Time taken for Updating Centroids = %f\n", global_time_taken_updating);
        printf("Total time taken = %f\n", global_time_taken_total);
    }



    // free dataset
    for (int i = 0; i < N; i++)
    {
        free(dataset[i]);
    }

    free(dataset);
    MPI_Finalize();

    return 0;
}

int importDataset(char *fname, int DIM, int N, double **dataset)
{

    FILE *fp = fopen(fname, "r");

    if (!fp)
    {
        printf("Unable to open file\n");
        return (1);
    }

    char buf[4096];
    int rowCnt = 0;
    int colCnt = 0;
    while (fgets(buf, 4096, fp) && rowCnt < N)
    {
        colCnt = 0;

        char *field = strtok(buf, ",");
        double tmp;
        sscanf(field, "%lf", &tmp);
        dataset[rowCnt][colCnt] = tmp;

        while (field)
        {
            colCnt++;
            field = strtok(NULL, ",");

            if (field != NULL)
            {
                double tmp;
                sscanf(field, "%lf", &tmp);
                dataset[rowCnt][colCnt] = tmp;
            }
        }
        rowCnt++;
    }

    fclose(fp);
    return 0;
}

