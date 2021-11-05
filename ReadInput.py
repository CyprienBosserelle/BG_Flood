# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
#%% Configuration
import re;

BG_ReadInput="src/ReadInput.cu"
BG_Params_h="src/Param.h"
BG_Forcing_h="src/Forcing.h"
ParamListFile="ParametersList-py.md"

#%% Getting the list of the available entry keys
# Get the parameter keys
keys=[]
ref_name=[]
Params=[]
Forcings=[]
F=open(BG_ReadInput,'r')
lines = F.readlines()
F.close()

#line='nodvWOONN parameterstr = TATAT  ;'
#toto=re.search('(^\s*parameterstr\s*=\s*(\"\s*\w+\s*\")(?=\s*;\s*$))', line)


#p = re.compile(r'\.*parameterstr = (?:\w+)\.*')

#with open(BG_ReadInput, 'r') as f:
#    while (line := f.readline().rstrip()):
#        print(line)


#m = re.findall("\.*parameterstr = (\w+)\.*", line, re.IGNORECASE)
#if m:
#    key=m[0];

for line in lines:
    if (('param.' in line) or ('XParam.' in line)):
        Params.append(re.findall("\.*(param.|XParam.)(\w+)\.*", line, re.IGNORECASE)[0][1])
    if (('forcing.' in line) or ('XForcing.' in line)):
        Forcings.append(re.findall("\.*(forcing.|XForcing.)(\w+)\.*", line, re.IGNORECASE)[0][1])
    line=re.sub(r"[\t\s]*","",line)
    if ('parameterstr=' in line):
        key_loc=re.findall("parameterstr=\"(\w+)\"\.*", line, re.IGNORECASE)[0]
        if not (key_loc in keys):
            keys.append(key_loc)
            ref_name.append(key_loc)
    elif ('paramvec=' in line):
        key_loc=re.findall(".*paramvec={(.*)}.*", line, re.IGNORECASE)[0]
        if not (key_loc in keys):
            keys.append(key_loc)
            ref_name.append(re.findall('\"(\w+)\".*',key_loc,re.IGNORECASE)[0])

myParams=list(set(Params))
myForcings=list(set(Forcings))

#%% Getting the information from the others files

# Get the parameters variables
Default=['DD']*len(keys)
Example=['EE']*len(keys)
Comment=['CC']*len(keys)
Domain=['NN']*len(keys)
P=open(BG_Params_h,'r')
P_lines = P.readlines()
F=open(BG_Forcing_h,'r')
F_lines = F.readlines()
F.close()
P.close() 


##Creation of the variables for the tables
for ind in range(len(ref_name)):
    refword=ref_name[ind]
    if (refword in myParams):
        Domain[ind]='Param'
        com=[]
        for line in P_lines:
            found=re.findall(rf"\.*{re.escape(refword)}\s*=(.*);\s*(\/\/)*\s*(.*)" , line)
            if len(found) > 0:
                com=found[0]
        if com:
            Comment[ind]=com[2]
            Default[ind]=com[0]
    if (refword in myForcings):
        Domain[ind]='Forcing'
        com=[]
        for i in range(len(F_lines)):
            found=re.findall(rf"\.*{re.escape(refword)}\s*;", F_lines[i])
            if (len(found) > 0) and (re.search(r'\s*\/\*', F_lines[i+1])):
                j=1
                Default[ind]=''
                Example[ind]=''
                Comment[ind]=''
                while j > 0:
                    line=F_lines[i+j]
                    j=j+1
                    EXX=re.search(r'\.*Ex:\s*(.*)',line);
                    DEF=re.search(r'\.*Default:\s*(.*)',line);
                    ENDCOM=re.search(r'\.*\*\/\s*',line);
                    line=re.sub(r"[\t\n]*","",line)
                    line=line.replace("*/","")
                    line=line.replace("/*","")
                    if (EXX):
                        Example[ind] = Example[ind] + '<br>' + EXX[1]
                    elif (DEF):
                        Default[ind]=Default[ind] + '<br>' + DEF[1]
                    else:
                        if (re.search(r'\w',line)):
                            Comment[ind] = Comment[ind] + '<br>' + line
                    if (ENDCOM):
                        j=-1
        



#%% Creation of the output file

#Creating the mark-down file/table for the list of the user input parameters
Out=open(ParamListFile,'w')
Out.write('# Paramter and Forcing list for BG_Flood\n\n')
Out.write('BG_flood user interface consists in a text file, associating key words to user chosen parameters and forcings.\n')

#Creation of the Parameter table in MD
#####Paramters
Out.write('## List of the Parameters\' input\n\n')
Out.write('|_Reference_|_Keys_|_default_|_Explanation_|\n')
Out.write('|----|---|---|---|\n')

for ind in range(len(ref_name)):
    if (Domain[ind] == 'Param'):
        mystr= "|" + str(ref_name[ind]) + "|" + str(keys[ind]) + "|" + str(Default[ind]) + "|" + str(Comment[ind]) + "|\n"
        Out.write(mystr)
Out.write('---\n\n')

#Creation of the Parameter table in MD
#####Paramters
Out.write('## List of the Forcings\' inputs\n\n')
Out.write('|_Reference_|_Keys_|_default_|_Example_|_Explanation_|\n')
Out.write('|----|---|---|---|---|\n')

for ind in range(len(ref_name)):
    if (Domain[ind] == 'Forcing'):
        mystr= "|" + str(ref_name[ind]) + "|" + str(keys[ind]) + "|" + str(Default[ind][4:]) + "|" + str(Example[ind][4:]) + "|"+ str(Comment[ind][4:]) + "|\n"
        Out.write(mystr)
Out.write('---\n\n')

Out.close()



