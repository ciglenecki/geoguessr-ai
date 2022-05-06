#!/bin/bash
script_name="python3 src/preprocess_csv_create_polygons.py"
csv_file="data/external/data_huge.csv"

$script_name --csv $csv_file --spacing 0.5;
$script_name --csv $csv_file --spacing 0.45;
$script_name --csv $csv_file --spacing 0.4;
$script_name --csv $csv_file --spacing 0.3;
$script_name --csv $csv_file --spacing 0.25;
$script_name --csv $csv_file --spacing 0.23;
$script_name --csv $csv_file --spacing 0.2;
$script_name --csv $csv_file --spacing 0.18;
$script_name --csv $csv_file --spacing 0.17;