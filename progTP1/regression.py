# -*- coding: utf-8 -*-

import argparse
import numpy as np
import solution_regression as sr
import gestion_donnees as gd
import matplotlib.pyplot as plt


################################
# Execution en tant que script
#
# Tapper python3 regression.py 1 sin 20 20 0.3 10 0.001
#
# dans un terminal
################################


example = '''Exemples:

    python regression.py sin 20 20 0.3 10 0.001
    python regression.py tanh 100 20 0.0 1 0.000001 --sklearn
    python regression.py lineaire 50 50 0.5 3 0.0'''


def get_arguments():

    parser = argparse.ArgumentParser(
        description='Régression polynomiale.',
        epilog=example,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('modele_gen', choices=['lineaire', 'sin', 'tanh'],
                        default='sin',
                        help='Choix de la fonction générant les données.')
    parser.add_argument('nb_train', type=int, default=20,
                        help='Nombre de données d\'entraînement.')
    parser.add_argument('nb_test', type=int, default=20,
                        help='Nombre de données de test.')
    parser.add_argument('bruit', type=float, default=0.3,
                        help='Amplitude du bruit appliqué aux données.')
    parser.add_argument('M', type=int, default=10,
                        help='Degré du polynome de la fonction de base '
                        '(recherche d\'hyperparamètres lorsque M<0).')
    parser.add_argument('lamb', type=float, default=0.001,
                        help='Coefficient de régularisation (lambda).')
    parser.add_argument('--sklearn', action='store_true',
                        help='Utiliser sklearn plutôt que votre propre '
                             'implémentation.')
    return parser


def main():

    p = get_arguments()
    args = p.parse_args()

    skl = args.sklearn
    modele_gen = args.modele_gen
    nb_train = args.nb_train
    nb_test = args.nb_test
    bruit = args.bruit
    m = args.M
    lamb = args.lamb

    # Paramètres du générateur de données
    w = [0.3, 4.1]  # Parametres du modele generatif

    # Gestionnaire de données
    gestionnaire_donnees = gd.GestionDonnees(
        w, modele_gen, nb_train, nb_test, bruit)
    # Génération des données d'entraînement et de test
    [x_train, t_train, x_test, t_test] = gestionnaire_donnees.generer_donnees()

    # Entrainement du modele de regression
    regression = sr.Regression(lamb, m, skl)
    regression.entrainement(x_train, t_train, using_sklearn=skl)

    # Predictions sur les ensembles d'entrainement et de test
    predictions_train = regression.prediction(x_train)
    predictions_test = regression.prediction(x_test)

    # Calcul des erreurs
    erreurs_entrainement = regression.erreur(t_train, predictions_train)
    erreurs_test = regression.erreur(t_test, predictions_test)

    print("Erreur d'entraînement :", "%.2f" % erreurs_entrainement.mean())
    print("Erreur de test :", "%.2f" % erreurs_test.mean())

    # Affichage
    gestionnaire_donnees.afficher_donnees_et_modele(x_train, t_train, True)
    predictions_range = np.array(
        [regression.prediction(x) for x in np.arange(0, 1, 0.01)])
    gestionnaire_donnees.afficher_donnees_et_modele(
        np.arange(0, 1, 0.01), predictions_range, False)

    if m >= 0:
        plt.suptitle('Resultat SANS recherche d\'hyperparametres')
    else:
        plt.suptitle('Resultat AVEC recherche d\'hyperparametres')
    plt.show()


if __name__ == "__main__":
    main()
