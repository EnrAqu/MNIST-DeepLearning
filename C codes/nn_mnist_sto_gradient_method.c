/*************************************************************************

    TRAIN A PLAIN VANILLA NEURAL NETWORK 

    INPUT:   	binary file with double elements

    OUTPUT:     Neural net with stochastic gradient

*************************************************************************/

/* Libraries to import */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <math.h>

/* Information on Images */

int width = 28;
int num_train = 60000;
int num_test = 10000;
int dimension = 784; 

/* Information on Neural Network */

int no_in_nodes = 784;
int no_hidden = 100;
int no_out_nodes = 10;

int mini_batch = 32;
double learning_rate = 0.5;
int num_epoch = 5;
int epoch = 60000;

/* Exit function */
void Error_program (int num, char* array) { printf("%s \n Error num %d ! \n ", array, num); exit(num); }

/* To initialize the weights */
double RandomGenerator() { return ( (double)(rand()) + 1. )/( (double)(RAND_MAX) + 1. ); }    // return a normally distributed random number
double normalRandom() { double y1=RandomGenerator(); double y2=RandomGenerator(); return cos(2. * 3.14 * y2)*sqrt(-2. * log(y1)); }

/* For the label */
int max_argument(double aL[no_out_nodes]);

/* statistichs tools */
void reset_confusion(int confusion_matrix[no_out_nodes][no_out_nodes]);
void print_(double *vector, int lung);


int main(int argc, char const *argv[]) {

	int q, i, k, j, count, prevision, reality;
	FILE* fin;
	double **labels_train, **labels_test, **train, **test, cost;

	/* Reading the binary file and storing all the input in two lists: TRAIN and TEST! */
	char *path_binary_path = "../data/train_and_test_mnist.bin";
	fin = fopen (path_binary_path, "rb");
	if ( fin == NULL) Error_program (1, "File not opened!\n ");
	printf("The binary file has been opened\n");
	/* Memory allocated */
	if ((train = (double **)malloc(num_train * sizeof(double *))) == NULL) Error_program(2, "There is no memory for train");
	for (q = 0; q < num_train; ++q) {if ((*(train+q) = (double *)malloc(dimension * sizeof(double))) == NULL) Error_program(3, "There is no memory for all trains"); }
	if ((test = (double **)malloc(num_test * sizeof(double *))) == NULL) Error_program(4, "There is no memory for test");
	for (q = 0; q < num_test; ++q) {if ((*(test+q) = (double *)malloc(dimension * sizeof(double))) == NULL) Error_program(4, "There is no memory for all tests"); }
	if ((labels_train = (double **)malloc(num_train * sizeof(double *))) == NULL) Error_program(2, "There is no memory for train");
	for (q = 0; q < num_train; ++q) {if ((*(labels_train+q) = (double *)malloc(dimension * sizeof(double))) == NULL) Error_program(3, "There is no memory for all trains"); }
	if ((labels_test = (double **)malloc(num_test * sizeof(double *))) == NULL) Error_program(4, "There is no memory for test");
	for (q = 0; q < num_test; ++q) {if ((*(labels_test+q) = (double *)malloc(dimension * sizeof(double))) == NULL) Error_program(6, "There is no memory for all tests"); }

	/* Read file in order; in this case is: 
		1 -  train_modified 			 -->  	number of bites = num_train * dimension * sizeof(double)
		2 -  lab_train_modified			 -->  	number of bites = num_train * num_output * sizeof(double)
		3 -  test_modified 				 -->  	number of bites = num_test * dimension * sizeof(double)
		4 -  lab_test_modified 			 -->  	number of bites = num_test * num_output * sizeof(double) */
	
	for (i = 0; i < num_train; ++i) { if( fread(*(train + i), sizeof(double), dimension, fin) != dimension) Error_program(8, "Did not read the train elements!"); } 
	for (i = 0; i < num_train; ++i) { if( fread(*(labels_train + i), sizeof(double), no_out_nodes, fin) != no_out_nodes) Error_program(9, "Did not read the train labels!");}
	for (i = 0; i < num_test; ++i) { if( fread(*(test + i), sizeof(double), dimension, fin) != dimension) Error_program(10, "Did not read the test elements!");} 
	for (i = 0; i < num_test; ++i) { if( fread(*(labels_test + i), sizeof(double), no_out_nodes, fin) != no_out_nodes) Error_program(11, "Did not read the test labels!");}

	printf("\nBinary file finished; starting Neural Network: initialization of the weights...\n");
	fclose(fin);

	clock_t start = clock();

	char *bar = "-------------------------------------------------------";


	double a1[no_hidden], z1[no_hidden], delta_1[no_hidden];
	double aL[no_out_nodes], zL[no_out_nodes], delta_L[no_out_nodes];
	int confusion_matrix[no_out_nodes][no_out_nodes];
	
	/* Declaration value weights and biasis */
	double hid_mat[no_in_nodes][no_hidden], b1[no_hidden];
	double err_hidm[no_in_nodes][no_hidden], err_b1[no_hidden];
	double last_mat[no_hidden][no_out_nodes], bL[no_out_nodes];
	double err_last[no_hidden][no_out_nodes], err_bL[no_out_nodes];
	
	/* initialization */
	srand(time(NULL));
	for (int i = 0; i < no_in_nodes; ++i) for (int j = 0; j < no_hidden; ++j) {
		hid_mat[i][j] = normalRandom()/sqrt(no_in_nodes);
		err_hidm[i][j] = 0.;
	}
	for (int i = 0; i < no_hidden; ++i) { b1[i] = 0.; err_b1[i] = 0.; }

	for (int i = 0; i < no_hidden; ++i) for (int j = 0; j < no_out_nodes; ++j) {
		last_mat[i][j] = normalRandom()/sqrt(no_hidden);
		err_last[i][j] = 0.;
	}
	for (int i = 0; i < no_out_nodes; ++i) { bL[i] = 0.; err_bL[i] = 0.; }

	printf("Neural network with one hidden Layer con %d neurons,\n- Learning_rate = %.3f, \n- Number of epochs = %d, \n- Mini Batch = %d \n\n", no_hidden, learning_rate, num_epoch, mini_batch);


	
	printf("Starting the Training\n\n");

	for (int k = 0; k < num_epoch * epoch; ++k) {
		/* FEED FORWARD */
		// First Layer
		for(int j=0; j < no_hidden; ++j) {
	   		z1[j] = b1[j];
	   		for (int i = 0; i < no_in_nodes; ++i) {
	   			z1[j] += train[k % num_train][i] * hid_mat[i][j];
	   		}
	   		a1[j] = 1./(1. + exp(-z1[j]));
	    }
	    //print_(a1, 100);
	    // Second Layer
		for(int j=0; j < no_out_nodes; ++j) {
	   		zL[j] = bL[j];
	   		for (int i = 0; i < no_hidden; ++i) {
	   			zL[j] += a1[i] * last_mat[i][j];
	   		}
	   		aL[j] = 1./(1. + exp(-zL[j]));
	    }
	    //print_(aL, 10);
	    /* BACK PROPAGATION */
	    cost = 0.;
	    for (int i = 0; i < no_out_nodes; ++i) {
	    	delta_L[i] = aL[i] - labels_train[k % num_train][i];
	    	cost += 0.5 * (delta_L[i]) * (delta_L[i]);
	    	delta_L[i] *= (1. - aL[i]) * aL[i];
	    	err_bL[i] += delta_L[i];
	    	for (int j = 0; j < no_hidden; ++j) {
	    		err_last[j][i] += delta_L[i] * a1[j];
	    	}
	    }
	//	print_(&cost, 1);
	//	print_(delta_L, 10);
	    for (int i = 0; i < no_hidden; ++i) {
	    	delta_1[i] = 0.;
	    	for (int j = 0; j < no_out_nodes; ++j) {
	    		delta_1[i] += last_mat[i][j] * delta_L[j] ;
	    	}
	    	delta_1[i] *= (1. - a1[i]) * a1[i];
	    	err_b1[i] += delta_1[i];
	    	for (int j = 0; j < no_in_nodes; ++j) {
	    		err_hidm[j][i] += delta_1[i] * train[k % num_train][j];
	    	}
	    }
	//	print_(delta_1, 100);  
	/* UPGRADE */
	    if ((k+1) % mini_batch == 0) {
			for (int i = 0; i < no_in_nodes; ++i) for (int j = 0; j < no_hidden; ++j) {
				hid_mat[i][j] -= learning_rate * (err_hidm[i][j]/mini_batch);
				err_hidm[i][j] = 0.;
			}
			for (int i = 0; i < no_hidden; ++i) { b1[i] -= learning_rate * (err_b1[i]/mini_batch); err_b1[i] = 0.;}
	
			for (int i = 0; i < no_hidden; ++i) for (int j = 0; j < no_out_nodes; ++j) {
				last_mat[i][j] -= learning_rate * (err_last[i][j]/mini_batch);
				err_last[i][j] = 0.;
			}
			for (int i = 0; i < no_out_nodes; ++i) { bL[i] -= learning_rate * (err_bL[i]/mini_batch); err_bL[i] = 0.;}
		}
		if ((k+1)%epoch == 0) {
			printf("End of the epoch num %d \n", (int)((k+1)/epoch));
		}
	}

	printf("Training complete! Testing... \n");
	
	reset_confusion(confusion_matrix);
	for (int k = 0; k < num_test; ++k) {
		// First Layer
		for(int j = 0; j < no_hidden; ++j) {
	   		z1[j] = b1[j];
	   		for (int i = 0; i < no_in_nodes; ++i) {
	   			z1[j] += test[k][i] * hid_mat[i][j];
	   		}
	   		a1[j] = 1./(1. + exp(-z1[j]));
	    }
	    // Second Layer
		for(int j = 0; j < no_out_nodes; ++j) {
	   		zL[j] = bL[j];
	   		for (int i = 0; i < no_hidden; ++i) {
	   			zL[j] += a1[i] * last_mat[i][j];
	   		}
	   		aL[j] = 1./(1. + exp(-zL[j]));
	    }
	    prevision = max_argument(aL);
		reality = max_argument(labels_test[k]);
		confusion_matrix[prevision][reality] += 1;
	}
	
	int prec = 0;
	for (int i = 0; i < 10; ++i) {
		prec += confusion_matrix[i][i];
	}
	printf("%s\n", bar);
	printf("Confusion Matrix: Predictions = rows, results = columns\n");
	printf("%s\n", bar);
	for (k = 0; k < no_out_nodes; ++k) {
		printf("|  ");
		for (j = 0; j < no_out_nodes; ++j) {
			printf("%4d ", confusion_matrix[k][j]);
		}
		printf(" | \n");
	}
	printf("%s\n", bar);
	printf("\n");
	
	printf("Precision is %.3f \n", ((double)(prec))/100);

	clock_t end = clock();
    printf("Execution time =  %f seconds \n\n\n", ((double)(end - start)) / CLOCKS_PER_SEC);

	return 0;
}

/* Functions used */

void reset_confusion(int confusion_matrix[no_out_nodes][no_out_nodes]){
	for (int i = 0; i < no_out_nodes; ++i) {
		for (int j = 0; j < no_out_nodes; ++j) {
			confusion_matrix[i][j] = 0; } } }

void print_(double *vector, int lung){   /* It is necessary only to check some data */
	for (int i = 0; i < lung; ++i) {
		printf("%.15f \n", vector[i]);
	}
	printf("\n\n"); }

int max_argument(double aL[no_out_nodes]) {
	int position = 0;
	for (int i = 1; i < no_out_nodes; ++i) if (aL[i] >= aL[position]) { 
		position = i; }
	return position; }


