

# File BG\_Flood.cu



[**FileList**](files.md) **>** [**src**](dir_68267d1309a1af8e8297ef4c3efbcdba.md) **>** [**BG\_Flood.cu**](BG__Flood_8cu.md)

[Go to the source code of this file](BG__Flood_8cu_source.md)



* `#include "BG_Flood.h"`





































## Public Functions

| Type | Name |
| ---: | :--- |
|  int | [**main**](#function-main) (int argc, char \* argv) <br>_Main entry point for the BG\_Flood model executable._  |
|  int | [**mainwork**](#function-mainwork) ([**Param**](classParam.md) XParam, [**Forcing**](structForcing.md)&lt; float &gt; XForcing, [**Model**](structModel.md)&lt; T &gt; XModel, [**Model**](structModel.md)&lt; T &gt; XModel\_g) <br>_Main model setup and execution routine for BG\_Flood._  |




























## Public Functions Documentation




### function main 

_Main entry point for the BG\_Flood model executable._ 
```C++
int main (
    int argc,
    char * argv
) 
```





**Parameters:**


* `argc` Number of command-line arguments 
* `argv` Array of command-line argument strings 



**Returns:**

Status code (0 for success)


Main function This function is the entry point to the software The main function setups all the init of the model and then calls the mainloop to actually run the model


There are 3 main class storing information about the model: XParam (class [**Param**](classParam.md)), XModel (class [**Model**](structModel.md)) and XForcing (class [**Forcing**](structForcing.md)) Leading X stands for eXecution and is to avoid confusion between the class variable and the class declaration When running with the GPU there is also XModel\_g which is the same as XModel but with GPU specific pointers


This function does:
* Reads the inputs to the model
* Allocate memory on GPU and CPU
* Prepare and initialise memory and arrays on CPU and GPU
* Setup initial condition
* Adapt grid if require
* Prepare output file
* Run main loop
* Clean up and close




It coordinates CPU and GPU resources and calls the mainwork routine for model execution. 


        

<hr>



### function mainwork 

_Main model setup and execution routine for BG\_Flood._ 
```C++
template<class T>
int mainwork (
    Param XParam,
    Forcing < float > XForcing,
    Model < T > XModel,
    Model < T > XModel_g
) 
```





**Template parameters:**


* `T` Data type 



**Parameters:**


* `XParam` [**Model**](structModel.md) parameters 
* `XForcing` [**Forcing**](structForcing.md) data 
* `XModel` [**Model**](structModel.md) structure (CPU) 
* `XModel_g` [**Model**](structModel.md) structure (GPU) 



**Returns:**

Status code (0 for success)


This function performs the main setup and execution steps for the BG\_Flood model:
* Reads and verifies input/forcing data
* Initializes mesh and initial conditions
* Performs initial adaptation
* Sets up GPU resources
* Runs the main simulation loop (MainLoop)
* Handles test mode if specified




Integrates all major model components and coordinates CPU/GPU execution. 


        

<hr>

------------------------------
The documentation for this class was generated from the following file `src/BG_Flood.cu`

