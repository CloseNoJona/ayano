# Redes

`default.nnue` es la red que se integra en el ejecutable. En las mediciones se llama c1. Su
arquitectura es `(768x8hm -> 1024)x2 -> 8` y ocupa 12,6 MB.

Se entrenó con partidas de [Leela Chess Zero](https://lczero.org), en concreto con las de su
ejecución de prueba T80 de enero de 2024, ya convertidas y filtradas por otras personas. La receta
completa está en [training/README.md](../training/README.md).

Contains information from
[Leela Chess Zero training data, test80 2024, filtered (min-v2.v6)](https://huggingface.co/datasets/linrock/test80-2024),
which is made available here under the
[Open Database License (ODbL)](https://opendatacommons.org/licenses/odbl/1-0/).

La red de la 0.10.0, b1, tiene la misma arquitectura y se entrenó con datos del ciclo T91 de Leela.
Sigue en la etiqueta [v0.10.0](https://github.com/CloseNoJona/ayano/tree/v0.10.0/nets). Las
anteriores, de gen0 a gen4, tienen otra arquitectura y el motor actual las rechaza. Siguen en la
etiqueta [v0.9.1](https://github.com/CloseNoJona/ayano/tree/v0.9.1/nets).
