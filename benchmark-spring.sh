#!/usr/bin/env bash

report="spring-report"
results="spring-results.txt"

rm -r $report
rm -r $results

#java -jar ./spring-petclinic-3.3.0-SNAPSHOT.jar&
#id=$!
#sleep 10
#jmeter -n -t ./petclinic_test_plan.jmx -l $results -e -o $report
jmeter -n -t ./petclinic_test_plan.jmx -l $results -e -o $report
# kill $id
