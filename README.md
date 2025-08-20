# Extended ER4IO (AWS-S3) – PDSW

To better understand the I/O behavior of emerging workloads, provide a more comprehensive characterization of HPC systems, and enable a consistent scoring model in the future, we have adopted the classic Roofline model for I/O workload analysis. Our model offers a clear view of how close observed I/O performance is to peak performance and can also help identify performance bottlenecks.

**ERT4IO** (https://github.com/NeuwirthLab/ERT4IO) is a Python script that plots the I/O Roofline graph for applications and benchmarks. It was originally based on parsed text files from Darshan outputs. In this work, we extend ERT4IO to also parse data from different output formats.

## Usage

```sh
python roofline.py
```

Make sure you have the following Python packages installed:

- Python ≥ 3.8
- numpy  
- pandas  
- matplotlib  
- matplotlib-label-line
