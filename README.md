# backpropagation-neural-network
Exercice : Utilisation d'un réseau de neurones à rétropropagation avec une couche cachée pour le regroupement/classement.

Le réseau de neurones avec rétropropagation avec apprentissage supervisé est utilisé pour la résolution de problèmes non-linéaires. Ici, la fonction d'activation est la fonction sigmoide {0,1} et sa dérivée pour la rétropropagation.

Fondamentalement, nous pouvons utiliser des réseaux neuronaux pour tous problèmes tant que nous avons des caractéristiques (par exemple, des coordonnées x et y sur le plan 2D) et des étiquettes, des noms de catégorie encodés sous forme binaire.

Chaque critère/caractéristique est représenté par un neurone d'entrée.
Chaque résultat/classe possible est un neurone de sortie. On associe des valeurs binaires à chaque neurone.
En général, le nombre de neurones cachés varies entre le nombre de neurones en entrée et en sortie. Parfois, on doit ajouter davantage. C'est là que ça devient expériemental et que l'on doit faire quelques essaies.

Dans cet exercice, on cherche à apprendre à classer des variétés d'Iris parmi celles décrites dans la base de connaissances du site de l'Université de Californie, Irvine : https://archive.ics.uci.edu/ml/datasets/iris

Ici, il y a 4 attributs donc 4 neurones en entrée : 
 1. longueur du sépale en cm 
 2. largeur du sépale en cm 
 3. longueur du pétale en cm 
 4. largeur du pétale en cm

Et 3 classes, 3 neurones de sortie : 
 1. Iris Setosa = 1 - 0 - 0
 2. Iris Versicolour = 0 - 1 - 0
 3. Iris Virginica = 0 - 0 - 1

Nous avons 150 cas décrivant ces variétés, dont 50 chacunes.

Note : La difficulté est de déterminer le nombre de couches cachées nécessaires pour résoudre un problème. Cependant, il semble qu'une seule couche suffise pour la plupart des cas.

Note : Le problème de surapprentissage existe encore...

Note : Il est possible que parmi les centaines de cas appris, qu'il y ait quelques mauvaises interprétations par le système. Celui-ci fera donc des erreurs... Comme un humain ? Chaque apprentissage crée un réseau différent, il est important de bien tester les cas appris afin de conserver le/les réseau(x) adéquat(s).

Note : Il existe d'autres types de réseaux neuronaux tels que les réseaux neuronaux récurrents et les réseaux neuronaux convolutifs où la topologie diffère.
 
 ## Téléchargement
Vous pouvez téléchargez l'archives .zip du code ou bien utilisez git : 
$ git clone https://github.com/computer-science-club-ca/backpropagation-neural-network.git

## L'environnement
C'est un projet Java.

Installez Java 8 (JDK): http://www.oracle.com/technetwork/java/javase/downloads/jdk8-downloads-2133151.html

Installez Eclipse Java EE IDE for Web Developers (Version: Mars, http://www.eclipse.org/downloads/packages/eclipse-ide-java-ee-developers/mars2)

Importez le projet Java dans votre espace de travail Eclipse et lancez-le comme une application Java ! :)
