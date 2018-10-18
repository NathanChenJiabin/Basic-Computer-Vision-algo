#include <iostream>
#include <mpi.h>
#include <sys/stat.h>
#include "Image.h"
#include "Classifieur.h"

using namespace std;
using namespace cv;

double error_function(Classifieur c, int Xi, int label){
    if(c.result_of_classification(Xi) == label){
        return 0.0;
    }else{
        return 1.0;
    }
}

const auto INFITY = (1.0) * (1 << 21);

int main(int argc, char** argv)
{
    // MPI: rank and process number
    //MPI_Request request;
    MPI_Status status;
    int tag;
    MPI_Init(&argc, &argv);
    int rank=0; MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    int p=0;    MPI_Comm_size(MPI_COMM_WORLD, &p);
    int root = 0;
    tag = 442;

    // Test
    //image_print_line(argv[1],rank);
    //declaration some parameters constant
    const int total_train_pos_images = 818;
    const int total_train_neg_images = 4415;
    const int total_dev_pos_images = 818;
    const int total_dev_neg_images = 4415;
    const int total_test_pos_images = 818;
    const int total_test_neg_images = 4415;
    const int K = 1000; // nomber iteration of train
    const int N = 1000; // nomber iteration of boosting
    const double epsilon = 0.01; // learning rate

    //processuss train of classifeurs faibles
    int r = -1; // a random number
    Image* img;
    vector<int> features; // One feature extracted for a image
    vector<Classifieur> classifieurs; // all of weak classifieurs for this proc

    for(int k = 1; k<=K; k++){
        if(rank==root){
            // root proc generate a random number and then broadcast;
            r = rand() % (total_train_neg_images+total_train_pos_images) + 1; // r is from 1 to total number of train images
        }

        MPI_Bcast(&r, 1, MPI_INT, root, MPI_COMM_WORLD);

        if(r<=total_train_pos_images){
            img = new Image(r-1, "pos", "usr/local/INF442-2018/P5/app");
        }else{
            img = new Image(r-total_train_pos_images-1, "neg", "usr/local/INF442-2018/P5/app");
        }

        img->setIntegral_image(); // calculate integral image
        features = img->calculFeatures(rank, p); // extract features local for this proc

        if(k==1){
            //Initialization of classifeurs for the first iteration step
            for(int i = 0; i<features.size(); i++){
                Classifieur c(rank, i);
                classifieurs.push_back(c);
            }
        }
        //update the weights of all of classifieurs
        for(int i = 0; i<features.size(); i++){
            classifieurs[i].update(epsilon, features[i], img->getLabel());
        }
    }

    //Processus of boosting//
    //declaration some variables used in dev/test

    //vector<double> errors_of_each_step;
    vector<double> coeffients;
    vector<Classifieur> final__useful_classifieur;

    //all of variables for storage the images and features
    vector<Image> img_dev_pos;
    vector<vector<int> > features_dev_pos;
    vector<Image> img_dev_neg;
    vector<vector<int> > features_dev_neg;

    // load all of images from file "dev" for boosting
    for(int i = 0; i< (total_dev_neg_images+total_dev_pos_images); i++){
        if(i<total_dev_pos_images){
            Image img_dev(i, "pos", "usr/local/INF442-2018/P5/dev");
            img_dev.setIntegral_image();
            img_dev_pos.push_back(img_dev);
            features_dev_pos.push_back(img_dev.calculFeatures(rank, p));
        }else{
            Image img_dev(i-total_dev_pos_images, "neg", "usr/local/INF442-2018/P5/dev");
            img_dev.setIntegral_image();
            img_dev_neg.push_back(img_dev);
            features_dev_neg.push_back(img_dev.calculFeatures(rank, p));
        }
    }



    double recv_errors[p];
    for(int k = 0; k<p; k++){
        recv_errors[k] = 0.0;
    }

    double lambda[total_dev_neg_images+total_dev_pos_images];
    //Initialization for lambda
    for (double &k : lambda) {
        k = 1.0 / (double)(total_dev_neg_images+total_dev_pos_images);
    }


    for(int k = 1; k<=N; k++){

        double elocal_min = INFITY;
        int local_pos = 0;


        for(int i = 0; i< classifieurs.size(); i++){
            double e = 0.0; // error for one classifier

            for(int j = 1; j<=total_dev_neg_images+total_dev_pos_images; j++){
                if(j<=total_dev_pos_images){
                    e+= lambda[j-1] * error_function(classifieurs[i], features_dev_pos[j-1][i], 1 );
                }else{
                    e+= lambda[j-1] * error_function(classifieurs[i], features_dev_neg[j-total_dev_pos_images-1][i], -1 );
                }

            }

            if(e<elocal_min){
                elocal_min = e;
                local_pos = i;
            }
        }

        // "recv_count" parameter is the count of elements received per process, not the total summation of counts from all processes
        MPI_Gather(&elocal_min, 1, MPI_DOUBLE, &recv_errors, 1, MPI_DOUBLE, root, MPI_COMM_WORLD);

        int indice=-1; // indicate that which proc has the minimum error
        if(rank == root){
            double emin = recv_errors[0];

            indice = 0;
            for(int j = 1; j<p; j++){
                if(recv_errors[j]<emin){
                    emin = recv_errors[j];

                    indice = j;
                }
            }
            //errors_of_each_step.push_back(emin);

            for(int j = 1; j<p; j++){
                MPI_Send(&indice, 1, MPI_INT, j, tag+j, MPI_COMM_WORLD);
            }
        }else{
            MPI_Recv(&indice, 1, MPI_INT, root, tag+rank, MPI_COMM_WORLD, &status);
        }

        if(rank == indice){

            //calculer alpha
            double alpha = log((1-elocal_min)/elocal_min) / 2.0 ;
            classifieurs[local_pos].setCoeff(alpha);

            //ajouter ce classifeur utile dans candidate
            final__useful_classifieur.push_back(classifieurs[local_pos]);

            //update tous les lambda
            for(int j= 1; j<= total_dev_neg_images+total_dev_pos_images; j++){
                if(j<=total_dev_pos_images){
                    lambda[j-1] = lambda[j-1] * exp(-(classifieurs[local_pos].result_of_classification(features_dev_pos[j-1][local_pos]))*alpha);
                }else{
                    lambda[j-1] = lambda[j-1] * exp((classifieurs[local_pos].result_of_classification(features_dev_neg[j-total_dev_pos_images-1][local_pos]))*alpha);
                }

            }

            //normalisation
            double somme =0.0;
            for (double j : lambda) {
                somme+= j;
            }
            for (double &j : lambda) {
                j = j / somme ;
            }

        }

        //broadcast les lambda a tous les autres procs
        MPI_Bcast(&lambda, total_dev_neg_images+total_dev_pos_images, MPI_DOUBLE, indice, MPI_COMM_WORLD );

    }

    // Test
    double theta = 0.2;

    vector<Image> img_test_pos;
    vector<vector<int> > features_test_pos;
    vector<Image> img_test_neg;
    vector<vector<int> > features_test_neg;
    //load all of images in file "test"
    for(int i = 0; i< (total_test_neg_images+total_test_pos_images); i++){
        if(i<total_test_pos_images){
            Image img_test(i, "pos", "usr/local/INF442-2018/P5/test");
            img_test.setIntegral_image();
            img_test_pos.push_back(img_test);
            features_test_pos.push_back(img_test.calculFeatures(rank, p));
        }else{
            Image img_test(i-total_test_pos_images, "neg", "usr/local/INF442-2018/P5/test");
            img_test.setIntegral_image();
            img_test_neg.push_back(img_test);
            features_test_neg.push_back(img_test.calculFeatures(rank, p));
        }
    }


    vector<int> result_of_prediction_of_pos;
    vector<int> result_of_prediction_of_neg;
    double alpha_sum_local = 0.0;
    double alpha_sum_total = 0.0;

    for (auto &j : final__useful_classifieur) {
        alpha_sum_local+= j.getCoeff();
    }
    MPI_Reduce(&alpha_sum_local, &alpha_sum_total, 1, MPI_DOUBLE, MPI_SUM, root,
               MPI_COMM_WORLD);


    double pred_local_pos_tab[features_test_pos.size()];
    double pred_final_pos_tab[features_test_pos.size()];
    for(int k = 0; k<features_test_pos.size(); k++){
        pred_final_pos_tab[k] = 0.0;
    }
    for(int i = 0; i< features_test_pos.size(); i++){
        double pred_local = 0.0;

        //calcul pred_local
        for (auto &j : final__useful_classifieur) {
            pred_local+= j.getCoeff() * j.result_of_classification(features_test_pos[i][j.getIdposition()]);
        }
        pred_local_pos_tab[i] = pred_local;

    }
    //reduce
    //reduce pred_local to pred_final
    MPI_Reduce(&pred_local_pos_tab, &pred_final_pos_tab, total_test_pos_images, MPI_DOUBLE, MPI_SUM, root, MPI_COMM_WORLD);
    if(rank==root){
        for(int i = 0; i<total_test_pos_images; i++){
            //prediction final
            if(pred_final_pos_tab[i] >= theta * alpha_sum_total){
                result_of_prediction_of_pos.push_back(1);
            }else{
                result_of_prediction_of_pos.push_back(-1);
            }

        }
    }

    double pred_local_neg_tab[total_test_neg_images];
    double pred_final_neg_tab[total_test_neg_images];
    for(int i = 0; i< features_test_neg.size(); i++){
        double pred_local = 0.0;

        //calcul pred_local
        for (auto &j : final__useful_classifieur) {
            pred_local+= j.getCoeff() * j.result_of_classification(features_test_neg[i][j.getIdposition()]);
        }
        pred_local_neg_tab[i] = pred_local;

    }
    //reduce
    //reduce pred_local to pred_final
    MPI_Reduce(&pred_local_neg_tab, &pred_final_neg_tab, total_test_neg_images, MPI_DOUBLE, MPI_SUM, root, MPI_COMM_WORLD);
    if(rank==root){
        for(int i = 0; i<total_test_neg_images; i++){
            //prediction final
            if(pred_final_neg_tab[i] >= theta * alpha_sum_total){
                result_of_prediction_of_neg.push_back(1);
            }else{
                result_of_prediction_of_neg.push_back(-1);
            }

        }
    }

    //evaluation
    int false_positif = 0;
    int false_negatif = 0;
    int true_positif = 0;
    int true_negatif = 0;
    double taux_faux_negatif;
    double taux_faux_positif;
    if(rank==root){
        for(int i = 0; i<= result_of_prediction_of_pos.size(); i++){
            if(result_of_prediction_of_pos[i] == 1){
                true_positif+=1;
            }else{
                false_negatif+=1;
            }
        }
        for(int i = 0; i<= result_of_prediction_of_neg.size(); i++){
            if(result_of_prediction_of_neg[i] == -1){
                true_negatif+=1;
            }else{
                false_positif+=1;
            }
        }

        taux_faux_positif = (double)false_positif / (false_positif+true_negatif) ;
        taux_faux_negatif = (double)false_negatif / (false_negatif+true_positif) ;

        cerr << "Taux de faux positif" << taux_faux_positif <<endl;
        cerr << "Taux de faux negatif" << taux_faux_negatif <<endl;


    }


    MPI_Finalize();

    return 0;
}
