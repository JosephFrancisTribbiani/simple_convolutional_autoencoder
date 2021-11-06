# simple_convolutional_autoencoder

Архитектура автоэнкодера представлена в таблице ниже. На вход подается изображение размера (3, 48, 48), которое энкодером преобразуется в вектор размера 512.

|Layer (type)|Output Shape|Param|
|---|---|---|
|Conv2d-1|[-1, 32, 48, 48]|896|
|BatchNorm2d-2|[-1, 32, 48, 48]|64|
|MaxPool2d-3|[-1, 32, 24, 24]|0|
|ELU-4|[-1, 32, 24, 24]|0|
|Conv2d-5|[-1, 64, 24, 24]|18,496|
|BatchNorm2d-6|[-1, 64, 24, 24]|128|
|MaxPool2d-7|[-1, 64, 12, 12]|0|
|ELU-8|[-1, 64, 12, 12]|0|
|Dropout2d-9|[-1, 64, 12, 12]|0|
|Conv2d-10|[-1, 128, 12, 12]|73,856|
|BatchNorm2d-11|[-1, 128, 12, 12]|256|
|MaxPool2d-12|[-1, 128, 6, 6]|0|
|ELU-13|[-1, 128, 6, 6]|0|
|Conv2d-14|[-1, 256, 6, 6]|295,168|
|BatchNorm2d-15|[-1, 256, 6, 6]|512|
|MaxPool2d-16|[-1, 256, 3, 3]|0|
|ELU-17|[-1, 256, 3, 3]|0|
|Conv2d-18|[-1, 500, 1, 1]|1,152,500|
|ELU-19|[-1, 500, 1, 1]|0|
|Flatten-20|[-1, 500]|0|
|Unflatten-21|[-1, 500, 1, 1]|0|
|ConvTranspose2d-22|[-1, 256, 3, 3]|1,152,256|
|ELU-23|[-1, 256, 3, 3]|0|
|Upsample-24|[-1, 256, 6, 6]|0|
|ConvTranspose2d-25|[-1, 128, 6, 6]|295,040|
|BatchNorm2d-26|[-1, 128, 6, 6]|256|
|ELU-27|[-1, 128, 6, 6]|0|
|Upsample-28|[-1, 128, 12, 12]|0|
|ConvTranspose2d-29|[-1, 64, 12, 12]|73,792|
|BatchNorm2d-30|[-1, 64, 12, 12]|128|
|ELU-31|[-1, 64, 12, 12]|0|
|Dropout2d-32|[-1, 64, 12, 12]|0|
|Upsample-33|[-1, 64, 24, 24]|0|
|ConvTranspose2d-34|[-1, 32, 24, 24]|18,464|
|BatchNorm2d-35|[-1, 32, 24, 24]|64|
|ELU-36|[-1, 32, 24, 24]|0|
|Upsample-37|[-1, 32, 48, 48]|0|
|ConvTranspose2d-38|[-1, 3, 48, 48]|867|
|BatchNorm2d-39|[-1, 3, 48, 48]|6|
|ELU-40|[-1, 3, 48, 48]|0|

Total params: 3,082,749  
Trainable params: 3,082,749  
Non-trainable params: 0

Автоэнкодер обучался на изображениях лиц людей из датасета `sklearn.datasets.fetch_lfw_people`. В качестве функции потерь выбрана `MSELoss()`, минимальное значение которой на этапе обучения составило 0.0015605360154533781.  
Графики изменения `loss` на тренировочной и тестовой выборках, а также график изменения `learning rate` представлены ниже.  

<img src='https://github.com/JosephFrancisTribbiani/simple_convolutional_autoencoder/blob/main/images/Training.png'></img>

Результаты работы автоэкодера представлены ниже. Первая строка - исходные изображения, вторая строка - восстановленные изображения из латентного представления исходных.
<img src='https://github.com/JosephFrancisTribbiani/simple_convolutional_autoencoder/blob/main/images/results.png'></img>
