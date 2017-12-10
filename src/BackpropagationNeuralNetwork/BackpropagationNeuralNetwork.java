package BackpropagationNeuralNetwork;

/**
 * R�seau de neurones avec r�tropropagation. Apprentissage supervis� utilis�
 * pour la r�solution de probl�mes non-lin�aires. Ici, la fonction d'activation
 * est la fonction sigmoide {0,1} et sa d�riv�e pour la r�tropropagation.
 * 
 * Fondamentalement, nous pouvons utiliser des r�seaux neuronaux pour tout
 * probl�me tant que nous avons des caract�ristiques (par exemple, des
 * coordonn�es x et y sur le plan 2D) et des �tiquettes, des noms de cat�gorie
 * encod�s sous forme binaire.
 * 
 * Chaque crit�re/caract�ristique est repr�sent� par un neurone d'entr�e.
 * 
 * Chaque r�sultat/classe possible est un neurone de sortie. On associe des
 * valeurs binaires � chaque neurone.
 * 
 * En g�n�ral, le nombre de neurones cach�s varies entre le nombre de neurones
 * en entr�e et en sortie. Parfois, on doit ajouter davantage. C'est l� que �a
 * devient exp�riemental et que l'on doit faire quelques essaies.
 *
 */
public class BackpropagationNeuralNetwork {
    // Taux d'apprentissage. La valeur devrait �tre entre 0.1 et 0.3
    public static final float LEARNING_RATE = 0.3f;
    // Inertie, divergence qui permet de sortir des minimum locaux
    public static final float MOMENTUM = 0.6f;
    // Nombre d'it�rations : epoch (Int�ressant de voir comment le syst�me
    // s'ajuste avec 10, 100, 1000 p�riodes)
    public static final int ITERATIONS = 10000;
    // Liste des connexions entre les neurones des diff�rentes couches("edge
    // weights")
    private Layer[] layers;

    /**
     * Constructeur du r�seau de neurones
     * 
     * @param inputSize
     *            (int), entier du nombre de neurones en entr�e
     * @param hiddenSize
     *            (int), entier du nombre de neurones cach�s
     * @param outputSize
     *            (int), entier du nombre de neurones en sortie
     */
    public BackpropagationNeuralNetwork(int inputSize, int hiddenSize, int outputSize) {
        layers = new Layer[2];
        layers[0] = new Layer(inputSize, hiddenSize);
        layers[1] = new Layer(hiddenSize, outputSize);
    }

    /**
     * Obtenir une couche de connexions ("edge weights")
     * 
     * @param index
     *            (int), entier de l'index de la couche de connexions
     * @return (Layer), la couche de connexions
     */
    public Layer getLayer(int index) {
        return layers[index];
    }

    /**
     * Processus qui pousse l'activit� neuronale de l'entr�e vers la sortie
     * 
     * @param input
     *            (float[]), une liste de valeur en entr�e
     * @return activations (float[]), la liste des valeurs r�sultantes
     *         d'activation en sortie
     */
    public float[] run(float[] input) {
        float[] activations = input;
        for (int i = 0; i < layers.length; i++) {
            activations = layers[i].run(activations);
        }
        return activations;
    }

    /**
     * Processus d'apprentissage avec r�tropropagation qui repousse l'activit�
     * neuronale de la sortie vers l'entr�e afin d'adapter les connexions.
     * 
     * @param input
     *            (float[]), liste des valeurs en entr�e
     * @param targetOutput
     *            (float[]), liste des valeurs attendues
     * @param learningRate
     *            (float), taux d'apprentissage. La valeur devrait �tre entre
     *            0.1 et 0.3
     * @param momentum
     *            (float), inertie qui sert � sortir des minimums locaux
     */
    public void train(float[] input, float[] targetOutput, float learningRate, float momentum) {
        // Liste des r�sultats des calculs dans le r�seau actuel
        float[] calculatedOutput = run(input);
        // Liste des erreurs pour chacune des donn�es d'entrainement
        float[] error = new float[calculatedOutput.length];
        // Calculer les erreurs
        for (int i = 0; i < error.length; i++) {
            error[i] = targetOutput[i] - calculatedOutput[i];
        }
        // r�tropropagation ("backpropagation"), obtenir les erreurs des valeurs
        // calcul�es
        for (int i = layers.length - 1; i >= 0; i--) {
            error = layers[i].train(error, learningRate, momentum);
        }
    }

    /**
     * Fonctions d'activation : Obtenir la fonction sigmoide
     * 
     * @param x
     *            (float), valeur � virgule flottante x
     * @return (float), valeur � virgule flottante de la sigmoide de x
     */
    public static float activationFunction(float x) {
        return (float) (1 / (1 + Math.exp(-x)));
    }

    /**
     * Fonctions d'activation de r�tropropagation : Obtenir la d�riv�e de la
     * fonction sigmoide
     * 
     * @param x
     *            (float), valeur � virgule flottante x
     * @return (float), valeur � virgule flottante de la d�riv�e de sigmoide de
     *         x
     */
    public static float activationFunctionBackPropagation(float x) {
        // Parce qu'on la sortie est la sigmoid(x), que la calcul est d�j�
        // en partie termin�, nous la calculons ainsi :
        return x * (1 - x);
    }
}
