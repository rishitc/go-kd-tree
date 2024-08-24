# Benchmarking Guide

## How to run the benchmarks?

* The benchmarks can be run by using the below command:
  * Ensure to fill in the placeholder values in the command with appropriate values before running it.

```bash
go test -benchtime=100x -tags trace -benchmem -run=^$ -bench ^<benchmark_function_name>$ github.com/rishitc/go-kd-tree/benchmarks/<competitor_folder_name>
```
