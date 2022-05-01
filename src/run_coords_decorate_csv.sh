#!/bin/bash
basename = python3 src/preprocess_csv_create_polygons.py
$basename --spacing 0.8;
$basename --spacing 0.7;
$basename --spacing 0.65;
$basename --spacing 0.6;
$basename --spacing 0.55;
$basename --spacing 0.5;
$basename --spacing 0.45;
$basename --spacing 0.4;
$basename --spacing 0.3;