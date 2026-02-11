# Dependencias del Generador NAV (C++)

Este documento detalla las librerías necesarias para compilar y ejecutar el generador de tiles NAV en C++.

## Librerías de Sistema

Las dependencias se pueden instalar en sistemas basados en Debian/Ubuntu con el siguiente comando:

```bash
sudo apt install build-essential cmake libosmium2-dev libgeos-dev \                 nlohmann-json3-dev libbz2-dev zlib1g-dev \                 libexpat1-dev libprotozero-dev
```

### Descripción de Componentes:

1.  **build-essential / cmake**: Herramientas básicas de compilación para C++.
2.  **libosmium2-dev**: Librería principal para el procesamiento de archivos OpenStreetMap (.pbf). Es extremadamente rápida y eficiente en memoria.
3.  **libgeos-dev**: Motor de geometría (Geometry Engine Open Source). Se encarga de las operaciones complejas como el recorte (clipping) de carreteras por el borde del tile y la fusión (union) de polígonos.
4.  **nlohmann-json3-dev**: Librería para parsear el archivo `features.json`.
5.  **libbz2-dev / zlib1g-dev / libexpat1-dev**: Dependencias internas de Osmium para descomprimir y leer el formato PBF.
6.  **libprotozero-dev**: Librería de bajo nivel para la lectura del formato Protocol Buffers usado en los archivos .pbf.

## Verificación de Instalación

Para verificar que el entorno está listo, puedes ejecutar la Fase 1 de compilación:

```bash
cd cpp_generator
mkdir build
cd build
cmake ..
make
```

Si `cmake` no muestra errores de "Package not found", el sistema está correctamente configurado.
