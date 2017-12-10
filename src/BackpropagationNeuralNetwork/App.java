package BackpropagationNeuralNetwork;

/**
 * Exercice : Utilisation d'un r�seau de neurones � r�tropropagation avec une
 * couche cach�e pour le regroupement/classement.
 * 
 * On cherche � lui apprendre � classer des vari�t�s d'Iris parmi celles
 * d�crites dans la base de connaissances du site de l'Universit� de Californie
 * : https://archive.ics.uci.edu/ml/datasets/iris
 * 
 * Ici, il y a 4 attributs donc 4 neurones en entr�e : 
 * 1. longeur du s�pale en cm 
 * 2. largeur du s�pale en cm 
 * 3. longueur du p�tale en cm 
 * 4. largeur du p�tale en cm
 * 
 * Et 3 classes, 3 neurones de sortie : 
 * 1. Iris Setosa = 1 - 0 - 0
 * 2. Iris Versicolour = 0 - 1 - 0
 * 3. Iris Virginica = 0 - 0 - 1
 * 
 * Nous avons 150 cas d�crivant ces vari�t�s, dont 50 chacunes.
 * 
 * Note : La difficult� est de d�terminer le nombre de couches cach�es
 * n�cessaires pour r�soudre un probl�me. Cependant, il semble qu'une seule
 * couche suffise pour la plupart des cas.
 * 
 * Note : Le probl�me de surapprentissage existe encore...
 * 
 * Note : Il est possible que parmi les centaines de cas appris, qu'il y ait
 * quelques mauvaises interpr�tations par le syst�me. Celui-ci fera donc des
 * erreurs... Comme un humain ? Chaque apprentissage cr�e un r�seau diff�rent,
 * il est important de bien tester les cas appris afin de conserver le/les
 * r�seau(x) ad�quat(s).
 * 
 * Note : Il existe d'autres types de r�seaux neuronaux tels que les r�seaux
 * neuronaux r�currents et les r�seaux neuronaux convolutifs o� la topologie
 * diff�re.
 * 
 */

public class App {

    public static void main(String[] args) throws Exception {

        // Au-lieu de lire le fichier de donn�es, les cas sont d�j� dans ces
        // tableaux. L'association entre les cas et leur r�sultat est bas� sur
        // l'ordre dans le tableau.
        // Base de connaissances
        float[][] trainingData = new float[][] { new float[] { 5.1f, 3.5f, 1.4f, 0.2f },
                new float[] { 4.9f, 3.0f, 1.4f, 0.2f }, new float[] { 4.7f, 3.2f, 1.3f, 0.2f },
                new float[] { 4.6f, 3.1f, 1.5f, 0.2f }, new float[] { 5.0f, 3.6f, 1.4f, 0.2f },
                new float[] { 5.4f, 3.9f, 1.7f, 0.4f }, new float[] { 4.6f, 3.4f, 1.4f, 0.3f },
                new float[] { 5.0f, 3.4f, 1.5f, 0.2f }, new float[] { 4.4f, 2.9f, 1.4f, 0.2f },
                new float[] { 4.9f, 3.1f, 1.5f, 0.1f }, new float[] { 5.4f, 3.7f, 1.5f, 0.2f },
                new float[] { 4.8f, 3.4f, 1.6f, 0.2f }, new float[] { 4.8f, 3.0f, 1.4f, 0.1f },
                new float[] { 4.3f, 3.0f, 1.1f, 0.1f }, new float[] { 5.8f, 4.0f, 1.2f, 0.2f },
                new float[] { 5.7f, 4.4f, 1.5f, 0.4f }, new float[] { 5.4f, 3.9f, 1.3f, 0.4f },
                new float[] { 5.1f, 3.5f, 1.4f, 0.3f }, new float[] { 5.7f, 3.8f, 1.7f, 0.3f },
                new float[] { 5.1f, 3.8f, 1.5f, 0.3f }, new float[] { 5.4f, 3.4f, 1.7f, 0.2f },
                new float[] { 5.1f, 3.7f, 1.5f, 0.4f }, new float[] { 4.6f, 3.6f, 1.0f, 0.2f },
                new float[] { 5.1f, 3.3f, 1.7f, 0.5f }, new float[] { 4.8f, 3.4f, 1.9f, 0.2f },
                new float[] { 5.0f, 3.0f, 1.6f, 0.2f }, new float[] { 5.0f, 3.4f, 1.6f, 0.4f },
                new float[] { 5.2f, 3.5f, 1.5f, 0.2f }, new float[] { 5.2f, 3.4f, 1.4f, 0.2f },
                new float[] { 4.7f, 3.2f, 1.6f, 0.2f }, new float[] { 4.8f, 3.1f, 1.6f, 0.2f },
                new float[] { 5.4f, 3.4f, 1.5f, 0.4f }, new float[] { 5.2f, 4.1f, 1.5f, 0.1f },
                new float[] { 5.5f, 4.2f, 1.4f, 0.2f }, new float[] { 4.9f, 3.1f, 1.5f, 0.1f },
                new float[] { 5.0f, 3.2f, 1.2f, 0.2f }, new float[] { 5.5f, 3.5f, 1.3f, 0.2f },
                new float[] { 4.9f, 3.1f, 1.5f, 0.1f }, new float[] { 4.4f, 3.0f, 1.3f, 0.2f },
                new float[] { 5.1f, 3.4f, 1.5f, 0.2f }, new float[] { 5.0f, 3.5f, 1.3f, 0.3f },
                new float[] { 4.5f, 2.3f, 1.3f, 0.3f }, new float[] { 4.4f, 3.2f, 1.3f, 0.2f },
                new float[] { 5.0f, 3.5f, 1.6f, 0.6f }, new float[] { 5.1f, 3.8f, 1.9f, 0.4f },
                new float[] { 4.8f, 3.0f, 1.4f, 0.3f }, new float[] { 5.1f, 3.8f, 1.6f, 0.2f },
                new float[] { 4.6f, 3.2f, 1.4f, 0.2f }, new float[] { 5.3f, 3.7f, 1.5f, 0.2f },
                new float[] { 5.0f, 3.3f, 1.4f, 0.2f }, new float[] { 7.0f, 3.2f, 4.7f, 1.4f },
                new float[] { 6.4f, 3.2f, 4.5f, 1.5f }, new float[] { 6.9f, 3.1f, 4.9f, 1.5f },
                new float[] { 5.5f, 2.3f, 4.0f, 1.3f }, new float[] { 6.5f, 2.8f, 4.6f, 1.5f },
                new float[] { 5.7f, 2.8f, 4.5f, 1.3f }, new float[] { 6.3f, 3.3f, 4.7f, 1.6f },
                new float[] { 4.9f, 2.4f, 3.3f, 1.0f }, new float[] { 6.6f, 2.9f, 4.6f, 1.3f },
                new float[] { 5.2f, 2.7f, 3.9f, 1.4f }, new float[] { 5.0f, 2.0f, 3.5f, 1.0f },
                new float[] { 5.9f, 3.0f, 4.2f, 1.5f }, new float[] { 6.0f, 2.2f, 4.0f, 1.0f },
                new float[] { 6.1f, 2.9f, 4.7f, 1.4f }, new float[] { 5.6f, 2.9f, 3.6f, 1.3f },
                new float[] { 6.7f, 3.1f, 4.4f, 1.4f }, new float[] { 5.6f, 3.0f, 4.5f, 1.5f },
                new float[] { 5.8f, 2.7f, 4.1f, 1.0f }, new float[] { 6.2f, 2.2f, 4.5f, 1.5f },
                new float[] { 5.6f, 2.5f, 3.9f, 1.1f }, new float[] { 5.9f, 3.2f, 4.8f, 1.8f },
                new float[] { 6.1f, 2.8f, 4.0f, 1.3f }, new float[] { 6.3f, 2.5f, 4.9f, 1.5f },
                new float[] { 6.1f, 2.8f, 4.7f, 1.2f }, new float[] { 6.4f, 2.9f, 4.3f, 1.3f },
                new float[] { 6.6f, 3.0f, 4.4f, 1.4f }, new float[] { 6.8f, 2.8f, 4.8f, 1.4f },
                new float[] { 6.7f, 3.0f, 5.0f, 1.7f }, new float[] { 6.0f, 2.9f, 4.5f, 1.5f },
                new float[] { 5.7f, 2.6f, 3.5f, 1.0f }, new float[] { 5.5f, 2.4f, 3.8f, 1.1f },
                new float[] { 5.5f, 2.4f, 3.7f, 1.0f }, new float[] { 5.8f, 2.7f, 3.9f, 1.2f },
                new float[] { 6.0f, 2.7f, 5.1f, 1.6f }, new float[] { 5.4f, 3.0f, 4.5f, 1.5f },
                new float[] { 6.0f, 3.4f, 4.5f, 1.6f }, new float[] { 6.7f, 3.1f, 4.7f, 1.5f },
                new float[] { 6.3f, 2.3f, 4.4f, 1.3f }, new float[] { 5.6f, 3.0f, 4.1f, 1.3f },
                new float[] { 5.5f, 2.5f, 4.0f, 1.3f }, new float[] { 5.5f, 2.6f, 4.4f, 1.2f },
                new float[] { 6.1f, 3.0f, 4.6f, 1.4f }, new float[] { 5.8f, 2.6f, 4.0f, 1.2f },
                new float[] { 5.0f, 2.3f, 3.3f, 1.0f }, new float[] { 5.6f, 2.7f, 4.2f, 1.3f },
                new float[] { 5.7f, 3.0f, 4.2f, 1.2f }, new float[] { 5.7f, 2.9f, 4.2f, 1.3f },
                new float[] { 6.2f, 2.9f, 4.3f, 1.3f }, new float[] { 5.1f, 2.5f, 3.0f, 1.1f },
                new float[] { 5.7f, 2.8f, 4.1f, 1.3f }, new float[] { 6.3f, 3.3f, 6.0f, 2.5f },
                new float[] { 5.8f, 2.7f, 5.1f, 1.9f }, new float[] { 7.1f, 3.0f, 5.9f, 2.1f },
                new float[] { 6.3f, 2.9f, 5.6f, 1.8f }, new float[] { 6.5f, 3.0f, 5.8f, 2.2f },
                new float[] { 7.6f, 3.0f, 6.6f, 2.1f }, new float[] { 4.9f, 2.5f, 4.5f, 1.7f },
                new float[] { 7.3f, 2.9f, 6.3f, 1.8f }, new float[] { 6.7f, 2.5f, 5.8f, 1.8f },
                new float[] { 7.2f, 3.6f, 6.1f, 2.5f }, new float[] { 6.5f, 3.2f, 5.1f, 2.0f },
                new float[] { 6.4f, 2.7f, 5.3f, 1.9f }, new float[] { 6.8f, 3.0f, 5.5f, 2.1f },
                new float[] { 5.7f, 2.5f, 5.0f, 2.0f }, new float[] { 5.8f, 2.8f, 5.1f, 2.4f },
                new float[] { 6.4f, 3.2f, 5.3f, 2.3f }, new float[] { 6.5f, 3.0f, 5.5f, 1.8f },
                new float[] { 7.7f, 3.8f, 6.7f, 2.2f }, new float[] { 7.7f, 2.6f, 6.9f, 2.3f },
                new float[] { 6.0f, 2.2f, 5.0f, 1.5f }, new float[] { 6.9f, 3.2f, 5.7f, 2.3f },
                new float[] { 5.6f, 2.8f, 4.9f, 2.0f }, new float[] { 7.7f, 2.8f, 6.7f, 2.0f },
                new float[] { 6.3f, 2.7f, 4.9f, 1.8f }, new float[] { 6.7f, 3.3f, 5.7f, 2.1f },
                new float[] { 7.2f, 3.2f, 6.0f, 1.8f }, new float[] { 6.2f, 2.8f, 4.8f, 1.8f },
                new float[] { 6.1f, 3.0f, 4.9f, 1.8f }, new float[] { 6.4f, 2.8f, 5.6f, 2.1f },
                new float[] { 7.2f, 3.0f, 5.8f, 1.6f }, new float[] { 7.4f, 2.8f, 6.1f, 1.9f },
                new float[] { 7.9f, 3.8f, 6.4f, 2.0f }, new float[] { 6.4f, 2.8f, 5.6f, 2.2f },
                new float[] { 6.3f, 2.8f, 5.1f, 1.5f }, new float[] { 6.1f, 2.6f, 5.6f, 1.4f },
                new float[] { 7.7f, 3.0f, 6.1f, 2.3f }, new float[] { 6.3f, 3.4f, 5.6f, 2.4f },
                new float[] { 6.4f, 3.1f, 5.5f, 1.8f }, new float[] { 6.0f, 3.0f, 4.8f, 1.8f },
                new float[] { 6.9f, 3.1f, 5.4f, 2.1f }, new float[] { 6.7f, 3.1f, 5.6f, 2.4f },
                new float[] { 6.9f, 3.1f, 5.1f, 2.3f }, new float[] { 5.8f, 2.7f, 5.1f, 1.9f },
                new float[] { 6.8f, 3.2f, 5.9f, 2.3f }, new float[] { 6.7f, 3.3f, 5.7f, 2.5f },
                new float[] { 6.7f, 3.0f, 5.2f, 2.3f }, new float[] { 6.3f, 2.5f, 5.0f, 1.9f },
                new float[] { 6.5f, 3.0f, 5.2f, 2.0f }, new float[] { 6.2f, 3.4f, 5.4f, 2.3f },
                new float[] { 5.9f, 3.0f, 5.1f, 1.8f } };

        // R�sultats attendus
        float[][] trainingResults = new float[][] { new float[] { 1f, 0f, 0f }, new float[] { 1f, 0f, 0f },
                new float[] { 1f, 0f, 0f }, new float[] { 1f, 0f, 0f }, new float[] { 1f, 0f, 0f },
                new float[] { 1f, 0f, 0f }, new float[] { 1f, 0f, 0f }, new float[] { 1f, 0f, 0f },
                new float[] { 1f, 0f, 0f }, new float[] { 1f, 0f, 0f }, new float[] { 1f, 0f, 0f },
                new float[] { 1f, 0f, 0f }, new float[] { 1f, 0f, 0f }, new float[] { 1f, 0f, 0f },
                new float[] { 1f, 0f, 0f }, new float[] { 1f, 0f, 0f }, new float[] { 1f, 0f, 0f },
                new float[] { 1f, 0f, 0f }, new float[] { 1f, 0f, 0f }, new float[] { 1f, 0f, 0f },
                new float[] { 1f, 0f, 0f }, new float[] { 1f, 0f, 0f }, new float[] { 1f, 0f, 0f },
                new float[] { 1f, 0f, 0f }, new float[] { 1f, 0f, 0f }, new float[] { 1f, 0f, 0f },
                new float[] { 1f, 0f, 0f }, new float[] { 1f, 0f, 0f }, new float[] { 1f, 0f, 0f },
                new float[] { 1f, 0f, 0f }, new float[] { 1f, 0f, 0f }, new float[] { 1f, 0f, 0f },
                new float[] { 1f, 0f, 0f }, new float[] { 1f, 0f, 0f }, new float[] { 1f, 0f, 0f },
                new float[] { 1f, 0f, 0f }, new float[] { 1f, 0f, 0f }, new float[] { 1f, 0f, 0f },
                new float[] { 1f, 0f, 0f }, new float[] { 1f, 0f, 0f }, new float[] { 1f, 0f, 0f },
                new float[] { 1f, 0f, 0f }, new float[] { 1f, 0f, 0f }, new float[] { 1f, 0f, 0f },
                new float[] { 1f, 0f, 0f }, new float[] { 1f, 0f, 0f }, new float[] { 1f, 0f, 0f },
                new float[] { 1f, 0f, 0f }, new float[] { 1f, 0f, 0f }, new float[] { 1f, 0f, 0f },
                new float[] { 0f, 1f, 0f }, new float[] { 0f, 1f, 0f }, new float[] { 0f, 1f, 0f },
                new float[] { 0f, 1f, 0f }, new float[] { 0f, 1f, 0f }, new float[] { 0f, 1f, 0f },
                new float[] { 0f, 1f, 0f }, new float[] { 0f, 1f, 0f }, new float[] { 0f, 1f, 0f },
                new float[] { 0f, 1f, 0f }, new float[] { 0f, 1f, 0f }, new float[] { 0f, 1f, 0f },
                new float[] { 0f, 1f, 0f }, new float[] { 0f, 1f, 0f }, new float[] { 0f, 1f, 0f },
                new float[] { 0f, 1f, 0f }, new float[] { 0f, 1f, 0f }, new float[] { 0f, 1f, 0f },
                new float[] { 0f, 1f, 0f }, new float[] { 0f, 1f, 0f }, new float[] { 0f, 1f, 0f },
                new float[] { 0f, 1f, 0f }, new float[] { 0f, 1f, 0f }, new float[] { 0f, 1f, 0f },
                new float[] { 0f, 1f, 0f }, new float[] { 0f, 1f, 0f }, new float[] { 0f, 1f, 0f },
                new float[] { 0f, 1f, 0f }, new float[] { 0f, 1f, 0f }, new float[] { 0f, 1f, 0f },
                new float[] { 0f, 1f, 0f }, new float[] { 0f, 1f, 0f }, new float[] { 0f, 1f, 0f },
                new float[] { 0f, 1f, 0f }, new float[] { 0f, 1f, 0f }, new float[] { 0f, 1f, 0f },
                new float[] { 0f, 1f, 0f }, new float[] { 0f, 1f, 0f }, new float[] { 0f, 1f, 0f },
                new float[] { 0f, 1f, 0f }, new float[] { 0f, 1f, 0f }, new float[] { 0f, 1f, 0f },
                new float[] { 0f, 1f, 0f }, new float[] { 0f, 1f, 0f }, new float[] { 0f, 1f, 0f },
                new float[] { 0f, 1f, 0f }, new float[] { 0f, 1f, 0f }, new float[] { 0f, 1f, 0f },
                new float[] { 0f, 1f, 0f }, new float[] { 0f, 1f, 0f }, new float[] { 0f, 0f, 1f },
                new float[] { 0f, 0f, 1f }, new float[] { 0f, 0f, 1f }, new float[] { 0f, 0f, 1f },
                new float[] { 0f, 0f, 1f }, new float[] { 0f, 0f, 1f }, new float[] { 0f, 0f, 1f },
                new float[] { 0f, 0f, 1f }, new float[] { 0f, 0f, 1f }, new float[] { 0f, 0f, 1f },
                new float[] { 0f, 0f, 1f }, new float[] { 0f, 0f, 1f }, new float[] { 0f, 0f, 1f },
                new float[] { 0f, 0f, 1f }, new float[] { 0f, 0f, 1f }, new float[] { 0f, 0f, 1f },
                new float[] { 0f, 0f, 1f }, new float[] { 0f, 0f, 1f }, new float[] { 0f, 0f, 1f },
                new float[] { 0f, 0f, 1f }, new float[] { 0f, 0f, 1f }, new float[] { 0f, 0f, 1f },
                new float[] { 0f, 0f, 1f }, new float[] { 0f, 0f, 1f }, new float[] { 0f, 0f, 1f },
                new float[] { 0f, 0f, 1f }, new float[] { 0f, 0f, 1f }, new float[] { 0f, 0f, 1f },
                new float[] { 0f, 0f, 1f }, new float[] { 0f, 0f, 1f }, new float[] { 0f, 0f, 1f },
                new float[] { 0f, 0f, 1f }, new float[] { 0f, 0f, 1f }, new float[] { 0f, 0f, 1f },
                new float[] { 0f, 0f, 1f }, new float[] { 0f, 0f, 1f }, new float[] { 0f, 0f, 1f },
                new float[] { 0f, 0f, 1f }, new float[] { 0f, 0f, 1f }, new float[] { 0f, 0f, 1f },
                new float[] { 0f, 0f, 1f }, new float[] { 0f, 0f, 1f }, new float[] { 0f, 0f, 1f },
                new float[] { 0f, 0f, 1f }, new float[] { 0f, 0f, 1f }, new float[] { 0f, 0f, 1f },
                new float[] { 0f, 0f, 1f }, new float[] { 0f, 0f, 1f }, new float[] { 0f, 0f, 1f },
                new float[] { 0f, 0f, 1f } };

        // R�seau de neurones avec 4 neurones en entr�e, 6 neurones cach�s et 3
        // neurones en sortie ... Exp�riementez avec un nombre variant de
        // neurones cach�s. :)
        BackpropagationNeuralNetwork backpropagationNeuralNetwork = new BackpropagationNeuralNetwork(4, 6, 3);

        // It�rer pour le nombre de p�riodes d'apprentissage...
        for (int iterations = 0; iterations < BackpropagationNeuralNetwork.ITERATIONS; iterations++) {

            // Apprentissage...
            for (int i = 0; i < trainingResults.length; i++) {
                backpropagationNeuralNetwork.train(trainingData[i], trainingResults[i],
                        BackpropagationNeuralNetwork.LEARNING_RATE, BackpropagationNeuralNetwork.MOMENTUM);
            }

            // Tester : afficher l'�volution � tous les 1000 tours...
            if ((iterations + 1) % 1000 == 0) {
                System.out.println("P�riode #" + iterations + "\n");
                for (int i = 0; i < trainingResults.length; i++) {
                    float[] data = trainingData[i];
                    float[] calculatedOutput = backpropagationNeuralNetwork.run(data);
                    System.out.println(data[0] + ", " + data[1] + ", " + data[2] + ", " + data[3] + " --> "
                            + Math.round(calculatedOutput[0]) + " - " + Math.round(calculatedOutput[1]) + " - "
                            + Math.round(calculatedOutput[2]));
                }
            }
        }
    }
}