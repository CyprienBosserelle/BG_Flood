

# Struct Model

**template &lt;class T&gt;**



[**ClassList**](annotated.md) **>** [**Model**](struct_model.md)


























## Public Attributes

| Type | Name |
| ---: | :--- |
|  std::vector&lt; double &gt; | [**OutputT**](#variable-outputt)  <br> |
|  std::map&lt; std::string, T \* &gt; | [**OutputVarMap**](#variable-outputvarmap)  <br> |
|  std::map&lt; std::string, std::string &gt; | [**Outvarlongname**](#variable-outvarlongname)  <br> |
|  std::map&lt; std::string, std::string &gt; | [**Outvarstdname**](#variable-outvarstdname)  <br> |
|  std::map&lt; std::string, std::string &gt; | [**Outvarunits**](#variable-outvarunits)  <br> |
|  T \* | [**Patm**](#variable-patm)  <br> |
|  T \* | [**TSstore**](#variable-tsstore)  <br> |
|  [**AdaptP**](struct_adapt_p.md) | [**adapt**](#variable-adapt)  <br> |
|  [**AdvanceP**](struct_advance_p.md)&lt; T &gt; | [**adv**](#variable-adv)  <br> |
|  [**BlockP**](struct_block_p.md)&lt; T &gt; | [**blocks**](#variable-blocks)  <br> |
|  [**BndblockP**](struct_bndblock_p.md)&lt; T &gt; | [**bndblk**](#variable-bndblk)  <br> |
|  T \* | [**cf**](#variable-cf)  <br> |
|  T \* | [**cl**](#variable-cl)  <br> |
|  T \* | [**datmpdx**](#variable-datmpdx)  <br> |
|  T \* | [**datmpdy**](#variable-datmpdy)  <br> |
|  [**EvolvingP\_M**](struct_evolving_p___m.md)&lt; T &gt; | [**evmax**](#variable-evmax)  <br> |
|  [**EvolvingP\_M**](struct_evolving_p___m.md)&lt; T &gt; | [**evmean**](#variable-evmean)  <br> |
|  [**EvolvingP**](struct_evolving_p.md)&lt; T &gt; | [**evolv**](#variable-evolv)  <br> |
|  [**EvolvingP**](struct_evolving_p.md)&lt; T &gt; | [**evolv\_o**](#variable-evolv_o)  <br> |
|  [**FluxP**](struct_flux_p.md)&lt; T &gt; | [**flux**](#variable-flux)  <br> |
|  [**FluxMLP**](struct_flux_m_l_p.md)&lt; T &gt; | [**fluxml**](#variable-fluxml)  <br> |
|  [**GradientsP**](struct_gradients_p.md)&lt; T &gt; | [**grad**](#variable-grad)  <br> |
|  T \* | [**hgw**](#variable-hgw)  <br> |
|  T \* | [**il**](#variable-il)  <br> |
|  [**TimeP**](struct_time_p.md)&lt; T &gt; | [**time**](#variable-time)  <br> |
|  T \* | [**wettime**](#variable-wettime)  <br> |
|  T \* | [**zb**](#variable-zb)  <br> |












































## Public Attributes Documentation




### variable OutputT 

```C++
std::vector<double> Model< T >::OutputT;
```




<hr>



### variable OutputVarMap 

```C++
std::map<std::string, T *> Model< T >::OutputVarMap;
```




<hr>



### variable Outvarlongname 

```C++
std::map<std::string, std::string> Model< T >::Outvarlongname;
```




<hr>



### variable Outvarstdname 

```C++
std::map<std::string, std::string> Model< T >::Outvarstdname;
```




<hr>



### variable Outvarunits 

```C++
std::map<std::string, std::string> Model< T >::Outvarunits;
```




<hr>



### variable Patm 

```C++
T* Model< T >::Patm;
```




<hr>



### variable TSstore 

```C++
T* Model< T >::TSstore;
```




<hr>



### variable adapt 

```C++
AdaptP Model< T >::adapt;
```




<hr>



### variable adv 

```C++
AdvanceP<T> Model< T >::adv;
```




<hr>



### variable blocks 

```C++
BlockP<T> Model< T >::blocks;
```




<hr>



### variable bndblk 

```C++
BndblockP<T> Model< T >::bndblk;
```




<hr>



### variable cf 

```C++
T* Model< T >::cf;
```




<hr>



### variable cl 

```C++
T* Model< T >::cl;
```




<hr>



### variable datmpdx 

```C++
T * Model< T >::datmpdx;
```




<hr>



### variable datmpdy 

```C++
T * Model< T >::datmpdy;
```




<hr>



### variable evmax 

```C++
EvolvingP_M<T> Model< T >::evmax;
```




<hr>



### variable evmean 

```C++
EvolvingP_M<T> Model< T >::evmean;
```




<hr>



### variable evolv 

```C++
EvolvingP<T> Model< T >::evolv;
```




<hr>



### variable evolv\_o 

```C++
EvolvingP<T> Model< T >::evolv_o;
```




<hr>



### variable flux 

```C++
FluxP<T> Model< T >::flux;
```




<hr>



### variable fluxml 

```C++
FluxMLP<T> Model< T >::fluxml;
```




<hr>



### variable grad 

```C++
GradientsP<T> Model< T >::grad;
```




<hr>



### variable hgw 

```C++
T* Model< T >::hgw;
```




<hr>



### variable il 

```C++
T* Model< T >::il;
```




<hr>



### variable time 

```C++
TimeP<T> Model< T >::time;
```




<hr>



### variable wettime 

```C++
T* Model< T >::wettime;
```




<hr>



### variable zb 

```C++
T* Model< T >::zb;
```




<hr>

------------------------------
The documentation for this class was generated from the following file `src/Arrays.h`

