## HOTSPOT

### Automatic mode:

-   command: java -jar bestgc.jar ./benchmark_gcs/benchmark_apps/dacapo-23.11-chopin.jar --args="spring -n 10 --no-pre-iteration-gc" --automatic --monitoringTime=50
-   Monotoring App with 50
-   java -jar -Xmx256m -Xms256m ./benchmark_gcs/benchmark_apps/dacapo-23.11-chopin.jar spring -n 10 --no-pre-iteration-gc
-   CpuAvgPercentage=72.48 Calculated weights: throughput weight=0.71, pause_time weight=0.29
-   Selected Heap Size: 512
-   Selecting best gc:
-   GC G1 score: 0.9999999701976776
-   GC Parallel score: 1.0947999674081803
-   GC Z score: 0.916199972331524

### Throughput Weight = 1

-   Command: java -jar bestgc.jar ./benchmark_gcs/benchmark_apps/dacapo-23.11-chopin.jar --args="spring -n 10 --no-pre-iteration-gc" --monitoringTime=50 --wt=1
-   Monotoring App with 50
-   java -jar -Xmx256m -Xms256m ./benchmark_gcs/benchmark_apps/dacapo-23.11-chopin.jar spring -n 10 --no-pre-iteration-gc
-   Selected Heap Size: 512
-   Selecting best gc:
-   GC G1 score: 1.0
-   GC Parallel score: 1.06
-   GC Z score: 1.27

### Throughput Weight = 0

-   Command: java -jar bestgc.jar ./benchmark_gcs/benchmark_apps/dacapo-23.11-chopin.jar --args="spring -n 10 --no-pre-iteration-gc" --monitoringTime=50 --wt=0
-   Monotoring App with 50
-   java -jar -Xmx256m -Xms256m ./benchmark_gcs/benchmark_apps/dacapo-23.11-chopin.jar spring -n 10 --no-pre-iteration-gc
-   Weights: throughput weight=0.0, pause_time weight=1.0
-   Selected Heap Size: 512
-   Selecting best gc:
-   GC G1 score: 1.0
-   GC Parallel score: 1.18
-   GC Z score: 0.05

## Graal

### Automatic mode:

-   command: java -jar bestgc.jar ./benchmark_gcs/benchmark_apps/dacapo-23.11-chopin.jar --args="spring -n 10 --no-pre-iteration-gc" --automatic --monitoringTime=50
-   Monotoring App with 50
-   java -jar -Xmx256m -Xms256m ./benchmark_gcs/benchmark_apps/dacapo-23.11-chopin.jar spring -n 10 --no-pre-iteration-gc
-   CpuAvgPercentage=73.09 Calculated weights: throughput weight=0.72, pause_time weight=0.28
-   Weights: throughput weight=0.7200000286102295, pause_time weight=0.2800000011920929
-   Selected Heap Size: 512
-   Selecting best gc:
-   GC G1 score: 1.0000000298023224
-   GC Parallel score: 1.0316000291705132
-   GC Z score: 0.918400036096573

### Throughput Weight = 1

-   Command: java -jar bestgc.jar ./benchmark_gcs/benchmark_apps/dacapo-23.11-chopin.jar --args="spring -n 10 --no-pre-iteration-gc" --monitoringTime=50 --wt=1
-   Monotoring App with 50
-   java -jar -Xmx256m -Xms256m ./benchmark_gcs/benchmark_apps/dacapo-23.11-chopin.jar spring -n 10 --no-pre-iteration-gc
-   Weights: throughput weight=1.0, pause_time weight=0.0
-   Selected Heap Size: 512
-   Selecting best gc:
-   GC G1 score: 1.0
-   GC Parallel score: 0.97
-   GC Z score: 1.26

### Throughput Weight = 0

-   Command: java -jar bestgc.jar ./benchmark_gcs/benchmark_apps/dacapo-23.11-chopin.jar --args="spring -n 10 --no-pre-iteration-gc" --monitoringTime=50 --wt=0
-   Monotoring App with 50
-   java -jar -Xmx256m -Xms256m ./benchmark_gcs/benchmark_apps/dacapo-23.11-chopin.jar spring -n 10 --no-pre-iteration-gc
-   Weights: throughput weight=0.0, pause_time weight=1.0
-   Selected Heap Size: 512
-   Selecting best gc:
-   GC G1 score: 1.0
-   GC Parallel score: 1.19
-   GC Z score: 0.04
