# VKLab_task

Код для обучения и запуска модели в файлу VAD_model.py
Пример запуска в файле colab_demo.ipynb
(гугл-диск последнее время вредничает, так что если веса не скачиваются, пишите[знал бы место лучше гугл-диска, загрузил бы туда])

<b>Сравнение с другим моделями на части https://www.openslr.org/resources/12/train-clean-100.tar.gz (accuracy):</b>
<br>(вообще наверное было бы адекватнее использовать precision-recall, так как меток 1 значительно больше)
<br>WebRTC: 0.9112798
<br>hmm-based model: 0.9108901536106
<br>My model: 0.9320501274095937 (я успел обучить ее только в течении двух эпох, так что результат мог бы быть и выше)

<br><b>Использованные материалы</b>:
<br>https://github.com/wiseman/py-webrtcvad
<br>https://github.com/eesungkim/Voice_Activity_Detector
<br>https://arxiv.org/pdf/1906.03588.pdf
<br>https://huggingface.co/speechbrain/vad-crdnn-libriparty
<br>(я хотел посмотреть различные архитектуры и если первые основаны на hidden markov models и gaussian mixture models, speechbrain - нейросетевая архитектура, но она видимо заточена под длительные аудиозаписи. В любом случае после нескольких попыток и следованию инструкции на сайте, качество получилось скудное, так что я не стал его вставлять)
Вообще я уже поздно вспомнил про https://docs.nvidia.com/deeplearning/nemo/user-guide/docs/en/stable/asr/speech_classification/models.html#marblenet-vad, но как-то времени читать про него не осталось, хотя наверное с этого стоило начать(
<br><b>Еще я хотел запустить и протестировать модель из этой статьи:</b>
<br>https://arxiv.org/pdf/2103.03529.pdf
<br>Но ребята не оставили гитхаба, так что я отказался от этой идеи

<br><b>P.S.</b> вообще мне в моей модели не нравится несколько вещей: я не супел пообучать ее на шумных аудиодорожках, мне кажется, если сделать fft для всей аудиодорожки и для начала там убрать паразитные частоты, то можно будет помимо всплесков шума лучше избавиться от постоянной части шума, присутствующей в дорожке. Можно было бы добавить attention. Это улучшило бы качество, но увеличило бы вычислительное времявообще можно было бы при желании и трансформера использовать, но тогда бы это игнорировало суть задачи: быстро выделить звук, чтобы не тратить ресурсы более глубоких моделей). Хотелось бы поэксперементировать с разным размером окна, да и имеющуюся модель хотелось бы побольше эпох пообучать, надо бы еще ввести взвешенный лосс, чтобы модель реже предсказывала наличие речи(так как речи в дорожках обычно больше, чем ее отсутствия).
