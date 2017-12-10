package BackpropagationNeuralNetwork;

/**
 * Réseau de neurones avec rétropropagation. Apprentissage supervisé utilisé
 * pour la résolution de problèmes non-linéaires. Ici, la fonction d'activation
 * est la fonction sigmoide {0,1} et sa dérivée pour la rétropropagation.
 * 
 * Fondamentalement, nous pouvons utiliser des réseaux neuronaux pour tout
 * problème tant que nous avons des caractéristiques (par exemple, des
 * coordonnées x et y sur le plan 2D) et des étiquettes, des noms de catégorie
 * encodés sous forme binaire.
 * 
 * Chaque critère/caractéristique est représenté par un neurone d'entrée.
 * 
 * Chaque résultat/classe possible est un neurone de sortie. On associe des
 * valeurs binaires à chaque neurone.
 * 
 * En général, le nombre de neurones cachés varies entre le nombre de neurones
 * en entrée et en sortie. Parfois, on doit ajouter davantage. C'est là que ça
 * devient expériemental et que l'on doit faire quelques essaies.
 *
 */
public class BackpropagationNeuralNetwork {
    // Taux d'apprentissage. La valeur devrait être entre 0.1 et 0.3
    public static final float LEARNING_RATE = 0.3f;
    // Inertie, divergence qui permet de sortir des minimum locaux
    public static final float MOMENTUM = 0.6f;
    // Nombre d'itérations : epoch (Intéressant de voir comment le système
    // s'ajuste avec 10, 100, 1000 périodes)
    public static final int ITERATIONS = 10000;
    // Liste des connexions entre les neurones des différentes couches("edge
    // weights")
    private Layer[] layers;

    /**
     * Constructeur du réseau de neurones
     * 
     * @param inputSize
     *            (int), entier du nombre de neurones en entrée
     * @param hiddenSize
     *            (int), entier du nombre de neurones cachés
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
     * Processus qui pousse l'activité neuronale de l'entrée vers la sortie
     * 
     * @param input
     *            (float[]), une liste de valeur en entrée
     * @return activations (float[]), la liste des valeurs résultantes
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
     * Processus d'apprentissage avec rétropropagation qui repousse l'activité
     * neuronale de la sortie vers l'entrée afin d'adapter les connexions.
     * 
     * @param input
     *            (float[]), liste des valeurs en entrée
     * @param targetOutput
     *            (float[]), liste des valeurs attendues
     * @param learningRate
     *            (float), taux d'apprentissage. La valeur devrait être entre
     *            0.1 et 0.3
     * @param momentum
     *            (float), inertie qui sert à sortir des minimums locaux
     */
    public void train(float[] input, float[] targetOutput, float learningRate, float momentum) {
        // Liste des résultats des calculs dans le réseau actuel
        float[] calculatedOutput = run(input);
        // Liste des erreurs pour chacune des données d'entrainement
        float[] error = new float[calculatedOutput.length];
        // Calculer les erreurs
        for (int i = 0; i < error.length; i++) {
            error[i] = targetOutput[i] - calculatedOutput[i];
        }
        // rétropropagation ("backpropagation"), obtenir les erreurs des valeurs
        // calculées
        for (int i = layers.length - 1; i >= 0; i--) {
            error = layers[i].train(error, learningRate, momentum);
        }
    }

    /**
     * Fonctions d'activation : Obtenir la fonction sigmoide
     * 
     * @param x
     *            (float), valeur à virgule flottante x
     * @return (float), valeur à virgule flottante de la sigmoide de x
     */
    public static float activationFunction(float x) {
        return (float) (1 / (1 + Math.exp(-x)));
    }

    /**
     * Fonctions d'activation de rétropropagation : Obtenir la dérivée de la
     * fonction sigmoide
     * 
     * @param x
     *            (float), valeur à virgule flottante x
     * @return (float), valeur à virgule flottante de la dérivée de sigmoide de
     *         x
     */
    public static float activationFunctionBackPropagation(float x) {
        // Parce qu'on la sortie est la sigmoid(x), que la calcul est déjà
        // en partie terminé, nous la calculons ainsi :
        return x * (1 - x);
    }
}
