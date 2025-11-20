

# Struct Forcing

**template &lt;class T&gt;**



[**ClassList**](annotated.md) **>** [**Forcing**](structForcing.md)


























## Public Attributes

| Type | Name |
| ---: | :--- |
|  [**AOIinfo**](classAOIinfo.md) | [**AOI**](#variable-aoi)  <br> |
|  [**DynForcingP**](structDynForcingP.md)&lt; T &gt; | [**Atmp**](#variable-atmp)  <br> |
|  std::vector&lt; [**StaticForcingP**](structStaticForcingP.md)&lt; T &gt; &gt; | [**Bathy**](#variable-bathy)  <br> |
|  [**DynForcingP**](structDynForcingP.md)&lt; T &gt; | [**Rain**](#variable-rain)  <br> |
|  [**DynForcingP**](structDynForcingP.md)&lt; T &gt; | [**UWind**](#variable-uwind)  <br> |
|  [**DynForcingP**](structDynForcingP.md)&lt; T &gt; | [**VWind**](#variable-vwind)  <br> |
|  std::vector&lt; [**bndsegment**](classbndsegment.md) &gt; | [**bndseg**](#variable-bndseg)  <br> |
|  [**bndparam**](classbndparam.md) | [**bot**](#variable-bot)  <br> |
|  std::vector&lt; [**StaticForcingP**](structStaticForcingP.md)&lt; T &gt; &gt; | [**cf**](#variable-cf)  <br> |
|  [**StaticForcingP**](structStaticForcingP.md)&lt; T &gt; | [**cl**](#variable-cl)  <br> |
|  std::vector&lt; [**deformmap**](classdeformmap.md)&lt; T &gt; &gt; | [**deform**](#variable-deform)  <br> |
|  [**StaticForcingP**](structStaticForcingP.md)&lt; T &gt; | [**il**](#variable-il)  <br> |
|  [**bndparam**](classbndparam.md) | [**left**](#variable-left)  <br> |
|  [**bndparam**](classbndparam.md) | [**right**](#variable-right)  <br> |
|  std::vector&lt; [**River**](classRiver.md) &gt; | [**rivers**](#variable-rivers)  <br> |
|  std::vector&lt; [**StaticForcingP**](structStaticForcingP.md)&lt; int &gt; &gt; | [**targetadapt**](#variable-targetadapt)  <br> |
|  [**bndparam**](classbndparam.md) | [**top**](#variable-top)  <br> |












































## Public Attributes Documentation




### variable AOI 

```C++
AOIinfo Forcing< T >::AOI;
```




<hr>



### variable Atmp 

```C++
DynForcingP<T> Forcing< T >::Atmp;
```




<hr>



### variable Bathy 

```C++
std::vector<StaticForcingP<T> > Forcing< T >::Bathy;
```




<hr>



### variable Rain 

```C++
DynForcingP<T> Forcing< T >::Rain;
```




<hr>



### variable UWind 

```C++
DynForcingP<T> Forcing< T >::UWind;
```




<hr>



### variable VWind 

```C++
DynForcingP<T> Forcing< T >::VWind;
```




<hr>



### variable bndseg 

```C++
std::vector<bndsegment> Forcing< T >::bndseg;
```




<hr>



### variable bot 

```C++
bndparam Forcing< T >::bot;
```




<hr>



### variable cf 

```C++
std::vector<StaticForcingP<T> > Forcing< T >::cf;
```




<hr>



### variable cl 

```C++
StaticForcingP<T> Forcing< T >::cl;
```




<hr>



### variable deform 

```C++
std::vector<deformmap<T> > Forcing< T >::deform;
```




<hr>



### variable il 

```C++
StaticForcingP<T> Forcing< T >::il;
```




<hr>



### variable left 

```C++
bndparam Forcing< T >::left;
```




<hr>



### variable right 

```C++
bndparam Forcing< T >::right;
```




<hr>



### variable rivers 

```C++
std::vector<River> Forcing< T >::rivers;
```




<hr>



### variable targetadapt 

```C++
std::vector<StaticForcingP<int> > Forcing< T >::targetadapt;
```




<hr>



### variable top 

```C++
bndparam Forcing< T >::top;
```




<hr>

------------------------------
The documentation for this class was generated from the following file `src/Forcing.h`

