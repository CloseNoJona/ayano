# Redes

`default.nnue` es la red que se integra en el ejecutable. En las mediciones se llama b1. Su
arquitectura es `(768x8hm -> 1024)x2 -> 8` y ocupa 12,6 MB.

Se entrenó con partidas de [Leela Chess Zero](https://lczero.org), en concreto con su ciclo T91, ya
convertidas y filtradas por otras personas. La receta completa está en
[training/README.md](../training/README.md).

Contains information from
[Filtered LC0 T91 (min-v2.v6)](https://huggingface.co/datasets/adamtwiss/t91-binpacks-filtered),
which is made available here under the
[Open Database License (ODbL)](https://opendatacommons.org/licenses/odbl/1-0/).

Las redes anteriores, de gen0 a gen4, tienen otra arquitectura y el motor actual las rechaza. Siguen
en la etiqueta [v0.9.1](https://github.com/CloseNoJona/ayano/tree/v0.9.1/nets).
