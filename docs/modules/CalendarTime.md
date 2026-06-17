# Calendar DateTime usage
By default BG_Flood uses number of seconds from a arbitrary default time to keet track of time. This is fine but using calendar time for input and output is very handy a lot more readable for humans.

You can use datetime string as input for TXT files as ```yyyy-mm-ddTHH:MM:SS``` as in ```year-month-day"T"hours:minutes:seconds``` also accepted ```yyyy/mm/ddTHH:MM:SS``` .


 
> Note the ```T``` is the key in BGFlood for identifying whether the input is a float (old format) or a datetime string.

You can still use the "old" float format. 

# reference time
The way that BG_Flood is working internally doesn't change so dealing with time string requires a **reference time**.

While the model doesn't need a reference time, it is preferable/good practice to set one up if using. That reference time can be specified in the parameter file. There are several options:

* Specify `reftime = ` as a date string.  Automatically `totaltime` will be set to `0` so `startime == reftime`
* Specify `totaltime=`; or `endtime` but no `reftime`. Automatically `reftime` is set to `totaltime` or `endtime` whichever comes first in the param file. 
* Do not specify `totaltime` or `endtime` or `reftime` in the param file. Then `reftime` will be obtained from the first specified string in boundaries, river, wind file. If none have datetime string forcing `reftime` will be set to `2000-01-01T00:00:00`



> IMPORTANT: 
> While you can let BG_Flood guess, it is recommended to set `starttime` and `endtime` to avoid surprises`

# Also for outputs
User can use `Toutput` instead of `outputtimestep` for range or single value(s) as long as they are separated by a `,`. Multiple ranges can be given and a mix of single values and range.

**Ranges are defined with pipe symbol `|`**

While `:` is commonly used to define ranges it wouldn't work for us. that is because we reserve `:` for separating time with specifying a date and there would be no way to distinguish between date and range. 
 
`Toutput=5.5,0|2.2|4,3.0`

### Time given in Toutput is either an absolute _date_ or a time relative to the model start time
`Toutput=3600.0` means 1 hour after start time!

### Supports time units
time can be given with a unit. 
`Toutput=5days,0|2.2s|4h,3.0min`

supported units are:
* __second__= { "seconds","second","sec","s" }; as 1sec
* __minute__ = { "minutes","minute","min","m" }; as 60sec
* __hour__ = { "hours","hour","hrs","hr","h" }; as 3600sec
* __day__ = { "days","day","d" }; as 24*3600sec
* __month__ = { "months","month","mths", "mth", "mon" }; as 3600.0 * 24.0 * 30.4375 sec
* __year__ = { "years","year","yrs", "yr", "y" }; as 3600.0 * 24.0 * 365.25

### Dates are also supported
as unique dates or ranges:
`Toutput=2020-01-01T00:00:00|2.2s|2020-01-01T00:00:04,2020-01-01T00:00:03`

means output from  2020-01-01T00:00:00 every 2.2s till 2020-01-01T00:00:04 and also 2020-01-01T00:00:03

> IMPORTANT: 
> regardless of the dates in Toutput the model always adds `starttime` and `endtime` to the list of outputs

>Note that range step can't be a date but instead need to be second or have a unit attached

## Example

Example of flow file or boundary file:
```
2020-01-01T00:00:00,1.000000
2020/01/02T01:00:00,1.000000
```
In your BG_param.txt

```
reftime = 2020-01-01T00:00:00
```


