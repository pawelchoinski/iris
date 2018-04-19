# -*- coding: utf-8 -*-
import pandas as pd
from sklearn.naive_bayes import GaussianNB
from sklearn import metrics
from sklearn.model_selection import train_test_split


class Dataset(object):
    """Klasa do zarządzania zbiorem danych"""
    #TODO obsługa wyjątkow, błędnych danych itd.
    def __init__(self, filename):
        self.get_data_from_csv(filename)
        pass

    def get_data_from_csv(self, filename):
        """Czyta dane z pliku csv i wrzuca do zmiennej data (pandas DataFrame)"""
        self.data = pd.read_csv(filename)

    def split_data(self, predicted_class_names, ratio, random_factor):
        # TODO dodać tworzenie z domyslnymi wartoscami
        """Dzieli zbiór na dane do treningu modelu i testowania modelu

        Parametry
        ----------
        predicted_class_names : str, nazwa kolumny z klasami/gatunakmi

        ratio : float, Wartości od 0.0 do 1.0, wskazuje jaka część danych ma trafić to zestawu testowego

        random_factor : int, pomocy przy dobraniu podobnego podziału procentowego
            danych treningowych i testowych
        """
        predicted_class_names = predicted_class_names
        attribute_names = list(self.data)
        attribute_names.remove(predicted_class_names)
        X = self.data[attribute_names].values
        y = self.data[predicted_class_names].values
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X, y, test_size=ratio, random_state=random_factor)

    def check_classes_ratio(self):
        """Sprawdza czy stosunek klas w danych treningowych i testowych jest zbliżony do
        stosunku klas w danych wyjściowych"""
        species = self.data['species'].unique()
        #TODO uprościć:
        ratio_orig = [[specie, len(self.data.loc[self.data['species'] == specie]), self.data.shape[0]] for specie in species]
        [a.append(a[1]/float(a[2])*100) for a in ratio_orig]
        ratio_train = [[specie, len(self.y_train[self.y_train[:] == specie]), len(self.y_train)] for specie in
                      species]
        [a.append(a[1] / float(a[2])*100) for a in ratio_train]
        ratio_test = [[specie, len(self.y_test[self.y_test[:] == specie]), len(self.y_test)] for specie in
                      species]
        [a.append(a[1] / float(a[2])*100) for a in ratio_test]
        #TODO uprościć:
        print("-"*80)
        print("Comparson of species ratio in original, test and train set:")
        for i in range(0, len(ratio_orig)):
            print("Original set {0}: {1} ({2:0.1f})%".format(ratio_orig[i][0], ratio_orig[i][1], ratio_orig[i][3]))
            print("Test set {0}: {1} ({2:0.1f})%".format(ratio_test[i][0], ratio_test[i][1], ratio_test[i][3]))
            print("Train set {0}: {1} ({2:0.1f})%".format(ratio_train[i][0], ratio_train[i][1], ratio_train[i][3]))
        print("-" * 80)

    def fix_ratio(self):
        """Do ustalenia wartosci random_factor, w przpadku gdy procentowy podział danych treingowych i testowych
        różni się znacznie od podziału danych oryginalnych"""
        #TODO zaimplementowac
        pass


class Reports(object):
    "Klasa do generowania raportów dokładności modelu"
    #TODO zaimplenetowac
    pass


def main():
    dataset = Dataset("dataset/iris.csv")       # otwiera plik z danymi i wrzuca go do obiektu dataset
    dataset.split_data('species', 0.20, 43)     # dzieli dataset na treingowy i testowy(20% danych)
    model = GaussianNB()                        # tworzy model NB
    model.fit(dataset.X_train, dataset.y_train.ravel())     # "trenuje" model danymi
    predict_train = model.predict(dataset.X_train)          # przewiduje wartości na podstawie danych treningowych
    dataset.check_classes_ratio()
    #sprawdzenie modelu z użyciem danych treningowych
    print("REPORTS:")
    print("Accuracy on training data: {0:.2f}".format(metrics.accuracy_score(dataset.y_train, predict_train)))

    # sprawdzenie modelu na danych testowych
    predict_test = model.predict(dataset.X_test)
    print("Accuracy on test data: {0:.2f}".format(metrics.accuracy_score(dataset.y_test, predict_test)))
    print("\nConfusion matrix - test data:")
    print(metrics.confusion_matrix(dataset.y_test,predict_test))
    print("\nClassification Report - test data")
    print(metrics.classification_report(dataset.y_test, predict_test))


if __name__ == '__main__':
    main()