# Sistema-Detector-de-Sonol-ncia-
Sistema de detecção de sonolência usando Mideapipe Face Mesh e OpenCV para calcular o aspectro da boca (MAR) e aspectro dos olhos (EAR) com calibração de limiares personalizados dos frames da base de dados YawDD e NTHU-DDD e classificação via modelos de aprendizado profundo. 

Projeto Final apresentado ao Departamento de Engenharia de Controle e Automação do
Centro Federal de Educação Tecnológica Celso Suckow da Fonseca campus Nova Iguaçu, como
parte dos requisitos necessários à obtenção do título de Bacharel em Engenharia de Controle
e Automação.

A sonolência ao volante é um dos principais fatores de risco para acidentes de
trânsito em rodovias brasileiras. Este trabalho propõe um sistema não intrusivo de
videomonitoramento para detecção de sonolência em motoristas, combinando pro
cessamento de imagens e aprendizado profundo. O pipeline desenvolvido emprega o
MediaPipe Face Mesh para extração de marcos faciais, a partir dos quais são calcu
ladas as razões de aspecto dos olhos (EAR) e da boca (MAR), utilizadas em uma
etapa de rotulagem automática por limiares adaptativos dos frames retirados das bases de dados YawDD e NTHU-DDD. O pré-processamento inclui
equalização adaptativa de histograma (CLAHE) para mitigar variações de ilumina
ção. Seis arquiteturas foram comparadas: uma CNN customizada com 5 camadas
convolucionais e cinco modelos pré-treinados no ImageNet — VGG16, ResNet50,
ResNet101, EfficientNet B0 e MobileNetV2 — com validação cruzada sobre a base
NTHU-DDD como domínio externo. A CNN customizada superou todos os modelos
pré-treinados na base YawDD, alcançando acurácia de 82,80%, F1-score de 82,07%
e AUC de 0,932. Na validação cruzada usando a NTHU-DDD, o VGG16 demonstrou maior robustez, re
gistrando o melhor AUC (0,767) e a menor queda de desempenho entre os modelos
avaliados. Os resultados indicam que arquiteturas compactas treinadas no domínio
específico superam modelos de maior complexidade em cenários fechados, enquanto
modelos pré-treinados apresentam maior capacidade de generalização.

Link da base de dados YawDD
https://ieee-dataport.org/open-access/yawdd-yawning-detection-dataset

Link da base de dados NTHU-DDD
https://www.kaggle.com/datasets/faisal7/nthuddd


