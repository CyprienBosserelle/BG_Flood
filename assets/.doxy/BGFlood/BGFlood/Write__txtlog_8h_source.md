

# File Write\_txtlog.h

[**File List**](files.md) **>** [**src**](dir_68267d1309a1af8e8297ef4c3efbcdba.md) **>** [**Write\_txtlog.h**](Write__txtlog_8h.md)

[Go to the documentation of this file](Write__txtlog_8h.md)


```C++

#ifndef WRITETXTLOG_H
#define WRITETXTLOG_H

#include "General.h"
#include "Param.h"

void log(std::string text);
void create_logfile();
void write_text_to_log_file(std::string text);
void SaveParamtolog(Param XParam);
void saveparam2netCDF(int ncid, int bgfid, Param XParam);
// End of global definition
#endif
```


