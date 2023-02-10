#!/bin/bash
# Script Bash per copiare le misure scaricate in una directory nota con nome noto

cd Downloads # mi sposto in Downloads dove si trova la cartella scaricata "perAbInf.tgz" 
tar xvzf perAbInf.tgz # estraggo la cartella "data" che contiene i tre set di misure a partire da "perAbInf.tgz"
cp -r data /home/ubuntu/Documents/Progetto  # copio i tre set di misure in una nuova cartella "Progetto" specificandone il path
exit

