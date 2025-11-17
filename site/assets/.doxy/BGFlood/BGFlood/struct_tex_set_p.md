

# Struct TexSetP



[**ClassList**](annotated.md) **>** [**TexSetP**](struct_tex_set_p.md)


























## Public Attributes

| Type | Name |
| ---: | :--- |
|  cudaArray \* | [**CudArr**](#variable-cudarr)  <br> |
|  cudaChannelFormatDesc | [**channelDesc**](#variable-channeldesc)   = `cudaCreateChannelDesc(32, 0, 0, 0, cudaChannelFormatKindFloat)`<br> |
|  float | [**dx**](#variable-dx)  <br> |
|  float | [**dy**](#variable-dy)  <br> |
|  float | [**nowvalue**](#variable-nowvalue)  <br> |
|  struct cudaResourceDesc | [**resDesc**](#variable-resdesc)  <br> |
|  cudaTextureObject\_t | [**tex**](#variable-tex)   = `0`<br> |
|  struct cudaTextureDesc | [**texDesc**](#variable-texdesc)  <br> |
|  bool | [**uniform**](#variable-uniform)  <br> |
|  float | [**xo**](#variable-xo)  <br> |
|  float | [**yo**](#variable-yo)  <br> |












































## Public Attributes Documentation




### variable CudArr 

```C++
cudaArray* TexSetP::CudArr;
```




<hr>



### variable channelDesc 

```C++
cudaChannelFormatDesc TexSetP::channelDesc;
```




<hr>



### variable dx 

```C++
float TexSetP::dx;
```




<hr>



### variable dy 

```C++
float TexSetP::dy;
```




<hr>



### variable nowvalue 

```C++
float TexSetP::nowvalue;
```




<hr>



### variable resDesc 

```C++
struct cudaResourceDesc TexSetP::resDesc;
```




<hr>



### variable tex 

```C++
cudaTextureObject_t TexSetP::tex;
```




<hr>



### variable texDesc 

```C++
struct cudaTextureDesc TexSetP::texDesc;
```




<hr>



### variable uniform 

```C++
bool TexSetP::uniform;
```




<hr>



### variable xo 

```C++
float TexSetP::xo;
```




<hr>



### variable yo 

```C++
float TexSetP::yo;
```




<hr>

------------------------------
The documentation for this class was generated from the following file `src/Forcing.h`

