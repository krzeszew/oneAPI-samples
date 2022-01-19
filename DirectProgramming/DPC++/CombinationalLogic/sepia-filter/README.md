# `Sepia-filter` Sample

The sepia filter is a program that converts a color image to a Sepia tone image, which is a monochromatic image with a distinctive Brown Gray color. The program works by offloading the compute intensive conversion of each pixel to Sepia tone and is implemented using DPC++ for CPU and GPU.

For comprehensive instructions see the [DPC++ Programming](https://software.intel.com/en-us/oneapi-programming-guide) and search based on relevant terms noted in the comments.

| Optimized for                     | Description
|:---                               |:---
| OS                                | Linux Ubuntu 18.04, Windows 10
| Hardware                          | Skylake with GEN9 or newer
| Software                          | Intel&reg; oneAPI DPC++/C++ Compiler
| What you will learn               | The Sepia Filter sample demonstrates the following using the Intel&reg; oneAPI DPC++/C++ Compiler <ul><li>Writing a custom device selector class</li><li>Offloading compute intensive parts of the application using both lamba and functor kernels</li><li>Measuring kernel execution time by enabling profiling</li></ul>
| Time to complete                  | 20 minutes

## Purpose

The sepia filter is a DPC++ application that accepts a color image as an input and converts it to a sepia tone image by applying the sepia filter coefficients to every pixel of the image. The sample demonstrates offloading the compute intensive part of the application, which is the processing of individual pixels to an accelerator with lambda and functor kernels' help.  The sample also demonstrates the usage of a custom device selector, which sets precedence for a GPU device over other available devices on the system.
The device selected for offloading the kernel is displayed in the output and the time taken to execute each of the kernels. The application also outputs a sepia tone image of the input image.

## Key implementation details

The basic DPC++ implementation explained in the code includes device selector, buffer, accessor, kernel, and command groups.  This sample also demonstrates a custom device selector's implementation by overwriting the SYCL device selector class, offloading computation using both lambda and functor kernels, and using event objects to time command group execution, enabling profiling.

## License

Code samples are licensed under the MIT license. See
[License.txt](https://github.com/oneapi-src/oneAPI-samples/blob/master/License.txt) for details.

Third party program Licenses can be found here: [third-party-programs.txt](https://github.com/oneapi-src/oneAPI-samples/blob/master/third-party-programs.txt)

## Building the Program for CPU and GPU

> Note: if you have not already done so, set up your CLI
> environment by sourcing the setvars script located in
> the root of your oneAPI installation.
>
> Linux Sudo: . /opt/intel/oneapi/setvars.sh
>
> Linux User: . ~/intel/oneapi/setvars.sh
>
> Windows: C:\Program Files(x86)\Intel\oneAPI\setvars.bat

### Include Files

The include folder is located at `%ONEAPI_ROOT%\dev-utilities\latest\include` on your development system.

### Running Samples in DevCloud

If running a sample in the Intel DevCloud, remember that you must specify the compute node (CPU, GPU, FPGA) and whether to run in batch or interactive mode. For more information, see the [Intel&reg; oneAPI Base Toolkit Get Started Guide](https://devcloud.intel.com/oneapi/get-started/base-toolkit/).

For specific instructions, jump to [Run the sample in the DevCloud](#run-on-devcloud)

### Using Visual Studio Code\*  (Optional)

You can use Visual Studio Code (VS Code) extensions to set your environment, create launch configurations,
and browse and download samples.

The basic steps to build and run a sample using VS Code include:

- Download a sample using the extension **Code Sample Browser for Intel oneAPI Toolkits**.
- Configure the oneAPI environment with the extension **Environment Configurator for Intel oneAPI Toolkits**.
- Open a Terminal in VS Code (**Terminal>New Terminal**).
- Run the sample in the VS Code terminal using the instructions below.

To learn more about the extensions and how to configure the oneAPI environment, see
[Using Visual Studio Code with Intel® oneAPI Toolkits](https://software.intel.com/content/www/us/en/develop/documentation/using-vs-code-with-intel-oneapi/top.html).

After learning how to use the extensions for Intel oneAPI Toolkits, return to this readme for instructions on how to build and run a sample.

### On a Linux\* System

Perform the following steps:

1. Build the program using the following `cmake` commands.

```bash
$ cd sepia-filter
$ mkdir build
$ cd build
$ cmake ..
$ make
```

2. Run the program

```bash
$ make run
```

### On a Windows\* System Using Visual Studio\* Version 2017 or Newer

- Build the program using VS2017 or VS2019
    - Right-click on the solution file and open using either VS2017 or VS2019 IDE.
    - Right-click on the project in Solution Explorer and select Rebuild.
    - From the top menu, select Debug -> Start without Debugging.

- Build the program using MSBuild
     - Open "x64 Native Tools Command Prompt for VS2017" or "x64 Native Tools Command Prompt for VS2019"
     - Run the following command: `MSBuild sepia-filter.sln /t:Rebuild /p:Configuration="Release"`

## Running the sample

### Application Parameters

The Sepia-filter application expects a png image as an input parameter. The application comes with some sample images in the input folder. One of these is specified as the default input image to be converted in the cmake file. The default output image is generated in the same folder as the application.

### Example Output

```text
Loaded image with a width of 3264, a height of 2448 and 3 channels
Running on Intel(R) Gen9
Submitting lambda kernel...
Submitting functor kernel...
Waiting for execution to complete...
Execution completed
Lambda kernel time: 10.5153 milliseconds
Functor kernel time: 9.99602 milliseconds
Sepia tone successfully applied to image:[input/silverfalls1.png]
```

### Running the sample in the DevCloud<a name="run-on-devcloud"></a>

1. Open a terminal on your Linux system.
2. Log in to DevCloud.

```bash
ssh devcloud
```

3. Download the samples.

```bash
git clone https://github.com/oneapi-src/oneAPI-samples.git
```

4. Change directories to the Sepia Filter sample directory.

```bash
cd ~/oneAPI-samples/DirectProgramming/DPC++/CombinationalLogic/sepia-filter
```

#### Build and run the sample in batch mode

The following describes the process of submitting build and run jobs to PBS.
A job is a script that is submitted to PBS through the qsub utility. By default, the qsub utility does not inherit the current environment variables or your current working directory. For this reason, it is necessary to submit jobs as scripts that handle the setup of the environment variables. In order to address the working directory issue, you can either use absolute paths or pass the `-d \<dir\>` option to qsub to set the working directory.

#### Create the Job Scripts

1. Create a `build.sh` script with your preferred text editor:

```bash
nano build.sh
```

2. Add this text into the `build.sh` file:

```bash
source /opt/intel/inteloneapi/setvars.sh > /dev/null 2>&1
mkdir build
cd build
cmake ..
make
```

3. Save and close the `build.sh` file.

4. Create a `run.sh` script with with your preferred text editor:

```bash
nano run.sh
```

5. Add this text into the `run.sh` file:

```bash
source /opt/intel/inteloneapi/setvars.sh > /dev/null 2>&1
cd build
make run
```

6. Save and close the `run.sh` file.

#### Build and run

Jobs submitted in batch mode are placed in a queue waiting for the necessary resources (compute nodes) to become available. The jobs will be executed on a first come basis on the first available node(s) having the requested property or label.

1. Build the sample on a gpu node.

```bash
qsub -l nodes=1:gpu:ppn=2 -d . build.sh
```

> Note: `-l nodes=1:gpu:ppn=2` (lower case L) is used to assign one full GPU node to the job.
>
> Note: The `-d .` is used to configure the current folder as the working directory for the task.

2. In order to inspect the job progress, use the qstat utility.

```bash
watch -n 1 qstat -n -1
```

>Note: The `watch -n 1` command is used to run `qstat -n -1` and display its results every second.

3. When the build job completes, there will be a `build.sh.oXXXXXX` file in the directory. After the build job completes, run the sample on a gpu node:

```bash
qsub -l nodes=1:gpu:ppn=2 -d . run.sh
```

4. When a job terminates, a couple of files are written to the disk:

    `<script_name>.sh.eXXXX`, which is the job stderr

    `<script_name>.sh.oXXXX`, which is the job stdout

    Here XXXX is the job ID, which gets printed to the screen after each qsub command.

5. Inspect the output of the sample.

```bash
cat run.sh.oXXXX
```

6. Remove the stdout and stderr files and clean-up the project files.

```bash
rm build.sh.*; rm run.sh.*; cd build; make clean
```

7. Disconnect from the Intel DevCloud.

```bash
exit
```

## Known Limitations

Due to a known issue in the Level0 driver, the sepia-filter fails with the default Level0 backend. A workaround is in place to enable the OpenCL backend.
