#!/bin/sh

echo $PWD

AIRFLOW_HOME=$PWD/airflow airflow standalone
