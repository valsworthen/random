Quelques notes sur le Kaggle proposé par Cdiscount

## Sur les données

Présentées sous forme de fichier BSON les données étaient LOURDES : 7 069 896 produits en train, 1 768 182 en test. Une contrainte supplémentaire était qu'une entrée du fichier BSON pouvait comporter plusieurs images ce qui complique la création des batchs : 12 371 508 images en train, 3 095 080 en test. Pour naviguer dans le fichier on peut s'inspirer du generateur proposé [ici](https://www.kaggle.com/humananalog/keras-generator-for-reading-directly-from-bson). Dans tous les cas il faut créer un générateur custom pour utiliser fit_generator avec un modèle.

Il est possible également de répartir les images dans des dossiers de train / validation pour utiliser flow_from_directory et utiliser un générateur de base de Keras. Cette répartition est TRÈS longue sur hdd et prend déjà +1h30 sur ssd.

Notes fun : les id des produits prennent des valeurs assez élevés à la fin du fichier ce qui a semblé ralentir _énormément_ le formatage en dataframe des prédictions. (10 min par batch de 64k --> 30min)

## Sur les modèles

Après une première submission réalisée en local je suis passé sur une g2.2x large d'AWS pur les deux dernières semaines du challenge. Coût : 100$. Déception quant à la performance qui ne semblait pas si supérieure à celle de mon PC.

### Custom model

Mon premier modèle à la mano a finalement plutôt bien performé (~40% Public LB) car il a été entraîné sur 2-3 epochs. **Images de 64x64x3** sur ce modèle. Une epoch prenait environ 12h et les prédictions 10h.

```python
    model = Sequential()
    model.add(Conv2D(64, kernel_size=(5, 5),padding = 'same',
                     activation='relu',
                     input_shape=input_shape))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(128, kernel_size=(5, 5),padding = 'same',
                     activation='relu'))
    model.add(MaxPooling2D(pool_size=(4, 4)))
    model.add(Flatten())
    model.add(Dense(1024, activation='relu'))
    model.add(Dense(num_classes, activation='softmax'))

    model.compile(loss=keras.losses.categorical_crossentropy,
                  optimizer=keras.optimizers.Adadelta(),
                  metrics=['accuracy'])
```

Une voie d'amélioration serait sûrement de passer à 180x180, d'augmenter la taille de la pénultième couche Dense et d'ajouter du Dropout.

### Modèle pré entraîné

Ce fut la **purge** ! J'ai voulu faire du transfer learning mais les kernels proposaient toujours d'utiliser fit_generator ce qui posait les problèmes énoncés au dessus. Quand j'ai eu réussi à unpack les images du BSON et à lancer du transfer learning je temps indiqué était de +24h par epoch.
J'ai donc simplement essayer de prédire grâce aux modèles suivants :
XceptionV2 de Miha Skalic : 38% (??)
InceptionV2 de Miha Skalic : 63%
InceptionV3 : pas été au bout car le noyau ne gérait pas bien les images.

Conclusion : j'aurais mieux fait de garder ma manière d'entrainer le modèle (**train_on_batch**) qui s'est révélée relativement correcte et presque plus simple à implémenter... Il fallait y ajouter un meilleur suivi de la progression avec tqdm.

J'aurais voulu tester VGG et ResNet mais j'ai laissé tomber. Aussi ResNet demande des inputs de **plus** de 180px. Comment faire ?
