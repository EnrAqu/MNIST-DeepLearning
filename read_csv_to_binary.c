/*************************************************************************

    READ .csv OF MNIST DATASET AND MAKE A BINARY FILE 

    INPUT:   	mnist_test.csv mnist_train.csv

    OUTPUT: 



*************************************************************************/

/* Libraries to import */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>


/* Information on Images - Global variables */

int width = 28;
int num_train = 60000;
int num_test = 10000;
int dimension = 784; 
int num_output = 10;

/* Declaration of functions, they are written under the main program */

int new_atoi (char *char_num);
void error_function(char *exit_string, int err_number);

/* Main element */

int main() {

/* Declaration of variables */
	FILE *fp;
	char *token, p, buff[4000];
	int counter, count = 0, error_count = 0;
	clock_t start = clock();
/* For the test data */
	char *path_file_test = "mnist_csv_test_train/mnist_test.csv"; 	// opening file
	error_count++;
	if ((fp = fopen(path_file_test,"r")) == NULL) error_function("Error reading test csv", error_count);
	// initialization all tools 
	int *lab_test, **test;
	error_count++;
	if ((lab_test = (int *)malloc(num_test * sizeof(int))) == NULL) error_function("Error allocating label test", error_count);
	error_count++;
	if ((test = (int **) malloc (num_test * sizeof(int*))) == NULL) error_function("Error allocating test pointer", error_count);
	error_count++;
    for (int q = 0; q < num_test; ++q) {
        *(test+q) = (int *)malloc(dimension * sizeof(int)); 
    	if (*(test + q) == NULL) error_function("Error allocating test+q pointer", error_count);
    }
    // Adding informations 
    count = 0;
	while (count < num_test ) { 
		p = getc(fp);
		*(lab_test + count) = p - '0';
		fgets(buff, 4000, fp);
		token = strtok(buff,",");
		*(*(test + count)) = new_atoi(token);
		counter = 0;
		while( counter < dimension ) {
			counter++;
			*( *(test + count) + counter) = new_atoi(token);
			token = strtok(NULL,",");
   		}
   		count++; }
    fclose(fp);		// closing file 

	/* NORMALIZATION OF THE DATA */
	error_count++;
	double **lab_test_modified, **test_modified;
	if ((lab_test_modified = (double**)malloc(num_test * sizeof(double*))) == NULL) error_function("Error allocating lab_test_modified", error_count);
	error_count++;
	for (int i = 0; i < num_test; ++i)  if ((*(lab_test_modified + i) = (double *)malloc(num_output * sizeof(double))) == NULL) error_function("Error allocating lab_test_modified + i", error_count);
	error_count++;
	if ((test_modified = (double**)malloc(num_test * sizeof(double*))) == NULL) error_function("Error allocating test_modified", error_count);
	error_count++;
	for (int i = 0; i < num_test; ++i) if ((*(test_modified + i) = (double *)malloc(dimension * sizeof(double))) == NULL) error_function("Error allocating test_modified + i", error_count);
	/* Modification of the data */
	for (int i = 0; i < num_test; ++i) {
		for (int j = 0; j < dimension; ++j) { 
			test_modified[i][j] = ((double)(test[i][j])/((double)255))*0.99 + 0.01; 
		}
		for (int j = 0; j < num_output; ++j) { 
			lab_test_modified[i][j] = 0.01; 
		}
		lab_test_modified[i][lab_test[i]] = 0.99; 
	}
	/* freeing unnecessary part */
	for (int i = num_test; i > 0; --i) { free( *(test + i )); }
	free(test); free(lab_test);
	printf("\n");

/* For the test data */
	
	char *path_file_train = "mnist_csv_test_train/mnist_train.csv"; 	// opening file
	error_count++;
	if ((fp = fopen(path_file_train,"r")) == NULL) error_function("Error reading train csv", error_count);
    // initialization all tools 
	int *lab_train, **train;
	error_count++;
	if ((lab_train = (int *)malloc(num_train * sizeof(int))) == NULL) error_function("Error allocating label train", error_count);
	error_count++;
	if ((train = (int **) malloc (num_train * sizeof(int*))) == NULL) error_function("Error allocating train pointer", error_count);
    error_count++;
	for (int q = 0; q < num_train; ++q) {
        *(train + q) = (int *)malloc(dimension * sizeof(int)); 
    	if (*(train + q) == NULL) error_function("Error allocating train+q pointer", error_count);
    }

    // Adding informations 
    count = 0;
	while (count < num_train ) { 
		p = getc(fp);
		*(lab_train + count) = p - '0';
		fgets(buff, 4000, fp);
		token = strtok(buff,",");
		*(*(train + count)) = new_atoi(token);
		counter = 0;
		while( counter < dimension ) {
			counter++;
			*( *(train + count) + counter) = new_atoi(token);
			token = strtok(NULL,",");
   		}
   		count++; }
    fclose(fp);	 // closing file 

    /* NORMALIZATION OF THE DATA */
    double **lab_train_modified, **train_modified;
   	error_count++;
	if ((lab_train_modified = (double**)malloc(num_train * sizeof(double*))) == NULL) error_function("Error allocating label train", error_count);
	error_count++;
	for (int i = 0; i < num_train; ++i)  if ((*(lab_train_modified + i) = (double *)malloc(num_output * sizeof(double))) == NULL) error_function("Error in label train malloc \n", error_count);
	error_count++;
	if ((train_modified = (double**)malloc(num_train * sizeof(double*))) == NULL) error_function("Error allocating train modified", error_count);
	error_count++;	
	for (int i = 0; i < num_train; ++i) if ((*(train_modified + i) = (double *)malloc(dimension * sizeof(double))) == NULL) error_function("Error in lab_train_modified + imalloc \n", error_count);
	/* Modification of the data */
	for (int i = 0; i < num_train; ++i) {
		for (int j = 0; j < dimension; ++j) { 
			train_modified[i][j] = ((double)(train[i][j])/((double)255))*0.99 + 0.01; 
		}
		for (int j = 0; j < num_output; ++j) { 
			lab_train_modified[i][j] = 0.01; 
		}
		lab_train_modified[i][lab_train[i]] = 0.99; 
	}
	/* freeing unnecessary part */
	for (int i = num_train; i > 0; --i) { free( *(train + i )); }
	free(train); free(lab_train);
	/* Check part*/
	int ciao = 60000-1;
	for (int i = 0; i < dimension; ++i) { 
		printf("%.4f ", train_modified[ciao][i]);
		if ((i+1)%28 == 0) { printf("\n"); }  
	}
	printf("\n");
	for (int i = 0; i < num_output; ++i) {
		printf("ciao %.3f  ", lab_train_modified[ciao][i]);
		if ((i+1)%28 == 0) { printf("\n"); }
	}

	
/* Now I have the modified data according to the rules of 		https://www.python-course.eu/neural_network_mnist.php     */
	
	printf("I start writing the Binary file \n");
	int i;
	FILE* fin;
	/* Open file
	In the binary file is important the order; in this case is: 
		1 -  train_modified 			 -->  	number of bites = num_train * dimension * sizeof(double)
		2 -  lab_train_modified			 -->  	number of bites = num_train * num_output * sizeof(double)
		3 -  test_modified 				 -->  	number of bites = num_test * dimension * sizeof(double)
		4 -  lab_test_modified 			 -->  	number of bites = num_test * num_output * sizeof(double)
	*/
	error_count++;
	if ((fin = fopen ("train_and_test_mnist.bin", "wb")) == NULL)  error_function("Error: The output binary data file cannot be opened", error_count);
	printf("\nBinary file created succesfully \n");   
	// Writing train files in blocks 
	error_count++;
	for(i=0; i < num_train; i++) if( fwrite(*(train_modified + i), sizeof(double), dimension, fin) != dimension) error_function("error when writing train", error_count);
	error_count++;
	for(i=0; i < num_train; i++) if( fwrite(*(lab_train_modified + i), sizeof(double), num_output, fin) != num_output) error_function("error when writing train labels", error_count);
	// Writing test files in blocks 
	error_count++;
	for(i=0; i < num_test; i++) if( fwrite(*(test_modified + i), sizeof(double), dimension, fin) != dimension) error_function("error when writing test", error_count);
	error_count++;
	for(i=0; i < num_test; i++) if( fwrite(*(lab_test_modified + i), sizeof(double), num_output, fin) != num_output) error_function("error when writing test labels", error_count);
	fclose(fin);
	
	printf("\nBinary file finished \n");

	clock_t end = clock();
	printf("Execution time =  %f seconds \n", ((double)(end - start)) / CLOCKS_PER_SEC);

	printf("\n");

	return 0;
}

void error_function(char *exit_string, int err_number) {
	printf("Error #%d, \n\n%s\n",err_number, exit_string);
	exit(1);
}

int new_atoi (char *char_num)  { 
	// It is the same of the atoi function pre-installed in C
	// (because is not working properly in this code for a reason that I don't know, I preferred to create a new one)
	// this function uses the property that char - '0' = int
	// in this case char_num has more than one number

    int num = 0; // Initialize the number that will be int version
    // Iterate through all characters of input string and add the new integer
    for (int i = 0; char_num[i] != '\0'; ++i) 
        num = num*10 + char_num[i] - '0';
    // return the number 
    return num; 
}




