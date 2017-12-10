package BackpropagationNeuralNetwork;

import java.util.Arrays;
import java.util.Random;

/**
 * Couche d'un r�seau de neurones
 */
public class Layer {
    // R�sultats calcul�s
    private float[] output;
    // Valeurs en entr�e
    private float[] input;
    // Poids des connexions
    private float[] weights;
    // Diff�rences/Changements pour "�duquer" les neurones
    private float[] deltaWeights;
    // G�n�rateur de nombres al�atoires
    private Random randomGenerator;

    /**
     * Constructeur
     * 
     * @param inputSize
     *            (int), nombre de neurones en entr�e
     * @param outputSize
     *            (int), nombre de neurones en sortie
     */
    public Layer(int inputSize, int outputSize) {
        output = new float[outputSize];
        // liste des neurones en entr�e + le neurone de biais "bias", voir
        // d�finition.
        input = new float[inputSize + 1];
        // il y a autant de connexions que le produit des neurones en entr�e
        // (+1) et celles en sortie
        weights = new float[(1 + inputSize) * outputSize];
        deltaWeights = new float[weights.length];
        this.randomGenerator = new Random();
        initWeights();
    }

    /**
     * Initialiser le poids des connexions avec des valeurs al�atoires entre
     * {-2,2}, parce que c'est l'intervalle habituellement utilis�e par les
     * scientifiques...
     */
    public void initWeights() {
        for (int i = 0; i < weights.length; i++) {
            weights[i] = (randomGenerator.nextFloat() - 0.5f) * 4f;
        }
    }

    /**
     * Lancer les calculs afin d'obtenir la liste des r�sultats apr�s la
     * fonction d'activation.
     * 
     * @param inputArray(float[]),
     *            liste des valeurs en entr�e
     * @return (float[]), 
     *            copie de la liste des valeurs r�sultantes
     */
    public float[] run(float[] inputArray) {
        // Copier l'inputArray dans input...
        System.arraycopy(inputArray, 0, input, 0, inputArray.length);
        input[input.length - 1] = 1; // bias
        int offset = 0;
        // Pour chacune des valeurs d'entr�e...
        for (int i = 0; i < output.length; i++) {
            // Faire la somme des produits du poids des connexions...
            for (int j = 0; j < input.length; j++) {
                output[i] += weights[offset + j] * input[j];
            }
            // Lancer la fonction d'activation pour cette somme...
            output[i] = BackpropagationNeuralNetwork.activationFunction(output[i]);
            offset += input.length;
        }
        return Arrays.copyOf(output, output.length);
    }

    /**
     * Apprentissage par r�tropropagation
     * 
     * @param error
     *            (float[]), les erreurs entre les valeurs attendues et celles
     *            calcul�es
     * @param learningRate
     *            (float), taux d'apprentissage. La valeur devrait �tre entre
     *            0.1 et 0.3
     * @param momentum
     *            (float), inertie qui sert � sortir des minimums locaux
     * @return nextError (float[]), liste des valeurs r�sultantes apr�s
     *         r�tropropagation
     */
    public float[] train(float[] error, float learningRate, float momentum) {
        int offset = 0;
        float[] nextError = new float[input.length];
        for (int i = 0; i < output.length; i++) {
            // Une seule couche cach�e, son delta ne change pas.
            float delta = error[i] * BackpropagationNeuralNetwork.activationFunctionBackPropagation(output[i]);
            for (int j = 0; j < input.length; j++) {
                // index des poids de connexion
                int previousWeightIndex = offset + j;
                nextError[j] = nextError[j] + weights[previousWeightIndex] * delta;
                // Calculer le gradient (la variation)
                float gradient = input[j] * delta;
                // Calculer le changement du poids � un moment t = gradient *
                // learningRate + momentum * le changement du poids de
                // l'it�ration pr�c�dente
                weights[previousWeightIndex] += gradient * learningRate + momentum * deltaWeights[previousWeightIndex];
                deltaWeights[previousWeightIndex] = gradient;
            }
            offset += input.length;
        }
        return nextError;
    }
}
