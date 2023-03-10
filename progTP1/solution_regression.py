# -*- coding: utf-8 -*-

#####
# VosNoms (Matricule) .~= À MODIFIER =~.
###

# NE PAS MODIFIER OU AJOUTER DE NOUVEAU IMPORTS
import numpy as np
import random
from sklearn.linear_model import Ridge


class Regression:
    def __init__(self, lamb: float, m: int = 1, using_sklearn: bool = False):
        self.lamb = lamb
        self.w = None
        self.using_sklearn = using_sklearn
        self.M = m

    def fonction_base_polynomiale(self, x):
        """
        Fonction de base qui projette la donnee x vers un espace polynomial tel
        que mentionne au chapitre 3. Si x est un scalaire, alors phi_x sera un
        vecteur à self.M dimensions : (x^1,x^2,...,x^self.M)
        Si x est un vecteur de N scalaires, alors phi_x sera un tableau 2D de
        taille NxM.

        NOTE : En mettant phi_x = x, on a une fonction de base lineaire qui
        fonctionne pour une regression lineaire.

        NOTE : Il s'agit d'un bon endroit pour inclure le biais.

        NOTE: Ne devrait nécéssiter qu'au plus une boucle, se fait facilement
        sans.
        """

        # AJOUTER CODE ICI
        phi_x = x
        return phi_x

    def recherche_hyperparametre(self, X, t):
        """
        Trouver la meilleure valeur pour l'hyperparamètre self.M (pour un
        lambda fixe donné en entrée).

        Option 1:
        Validation croisée de type "k-fold" avec k=10. La méthode array_split
        de numpy peut être utlisée pour diviser les données en "k" parties. Si
        le nombre de données en entrée N est plus petit que "k", k devient égal
        à N. Il est important de mélanger les données ("shuffle") avant de les
        sous-diviser en "k" parties.

        Option 2:
        Sous-échantillonage aléatoire avec ratio 80:20 pour Dtrain et Dvalid,
        avec un nombre de répétition k=10.

        Le resultat de la recherche est mis dans la variable self.M. Vous
        devrez assigner self.M lors de la recherche.

        X: vecteur de donnees
        t: vecteur de cibles
        """

        # AJOUTER CODE ICI
        print('M trouvé: {}'.format(self.M))

    def entrainement(self, X, t):
        """
        Entraîne la regression lineaire sur l'ensemble d'entraînement forme des
        entrees ``X`` (un tableau 2D Numpy, ou la n-ieme rangee correspond à
        l'entree x_n) et des cibles ``t`` (un tableau 1D Numpy ou le
        n-ieme element correspond à la cible t_n). L'entraînement doit
        utiliser le poids de regularisation specifie par ``self.lamb``.

        Cette methode doit assigner le champs ``self.w`` au vecteur
        (tableau Numpy 1D) de taille D+1, tel que specifie à la section 3.1.4
        du livre de Bishop.

        Lorsque using_sklearn=True, vous devez utiliser la classe "Ridge" de
        la librairie sklearn (voir
        http://scikit-learn.org/stable/modules/linear_model.html)

        Lorsque using_sklearn=Fasle, vous devez implementer l'equation 3.28 du
        livre de Bishop. Il est suggere que le calcul de ``self.w`` n'utilise
        pas d'inversion de matrice, mais utilise plutôt une procedure
        de resolution de systeme d'equations lineaires (voir np.linalg.solve).

        Aussi, la variable membre self.M sert à projeter les variables X vers
        un espace polynomiale de degre M (voir fonction
        self.fonction_base_polynomiale())

        NOTE IMPORTANTE : lorsque self.M < 0, il faut trouver la bonne valeur
        de self.M

        NOTE: Ne devrait nécéssiter aucune boucle.
        """
        # AJOUTER CODE ICI
        if self.M < 0:
            self.recherche_hyperparametre(X, t)

        phi_x = self.fonction_base_polynomiale(X)

        if self.using_sklearn:
            # AJOUTER CODE ICI
            self.w = None
        else:
            # AJOUTER CODE ICI
            self.w = None

    def prediction(self, x):
        """
        Retourne la prédiction de la régression pour une ou plusieurs entrées,
        representées par un tableau 1D Numpy ``x``.

        Cette méthode suppose que la methode ``entrainement()``
        a prealablement été appelée. Elle doit utiliser le champs ``self.w``
        afin de calculer la prediction y(x,w) (equation 3.1 et 3.3).

        NOTE: Ne nécéssite aucune boucle.

        NOTE: Se fait très bien sans condition particulière sur la taille
        de `x`.
        """
        # AJOUTER CODE ICI
        return 0.5

    @staticmethod
    def erreur(t, prediction):
        """
        Retourne la différence au carré entre la cible ``t`` et
        la prediction ``prediction``. Votre implémentation devrait gérer le
        calcul de l'erreur entre une seule donnée et sa cible et un vecteur
        de données et leurs cibles sans condition particulière.

        """

        # AJOUTER CODE ICI
        return 0.0
