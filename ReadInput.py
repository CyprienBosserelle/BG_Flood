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
# Get the parameter keys, as well as the list of variables associated to the
# Param and Forcing objects, and the reference names
keys=[]
ref_name=[]
Params=[]
Forcings=[]
R=open(BG_ReadInput,'r')
lines = R.readlines()
R.close()


for line in lines:
#    if not(re.findall('^\s*(//).*',line)):
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
        key_loc=re.sub(r"\""," ",key_loc)
        if not (key_loc in keys):
            keys.append(key_loc)
            ref_name.append(re.findall('(\w+) ,.*',key_loc,re.IGNORECASE)[0])

myParams=list(set(Params))
myForcings=list(set(Forcings))

#%% Getting the information from the others files

#Getting the information from the others files (Param.h / Forcing.h) and link
#keys with comment/Example and Default value 


# Get the parameters variables
Default=['']*len(keys)
Example=['']*len(keys)
Comment=['']*len(keys)
Domain=['NN']*len(keys)
P=open(BG_Params_h,'r')
P_lines = P.readlines()
F=open(BG_Forcing_h,'r')
F_lines = F.readlines()
F.close()
P.close() 


class InfoTable:
    def __init__(self):
        self.Reference=[]
        self.Keys=[]
        self.Default=[]
        self.Example=[]
        self.Comment=[]

ParamTable=InfoTable()
ForcingTable=InfoTable()
NonIdTable=InfoTable()


##Creation of the variables for the tables
for ind in range(len(ref_name)):
    refword=ref_name[ind]
    key=keys[ind]
    #print(refword)
    if (refword in myParams):
        print('Param')
        #Domain='Param'
        com=[]
        for i in range(len(P_lines)):
            line=P_lines[i]
            if 'std' in line:
                found=re.findall(rf"\s*std.*\s* {re.escape(refword)}(\.*);\s*(\/\/)*\s*(.*)" , line)
            else:
                found=re.findall(rf"\s*\w\s* {re.escape(refword)} \s*=*(.*);\s*(\/\/)*\s*(.*)" , line)
            #found=re.findall(rf"\.*{re.escape(refword)}\s*=*\s*([^\s]+)*;\s*(//)*\s*(.*)" , line)
            if len(found) > 0:
                com=found[0]
                Default=''
                Example=''
                Comment=''
                if (re.search(r'\s*\/\*', P_lines[i+1])):
                    j=1
                    while j > 0:
                        #print(j)
                        line=P_lines[i+j]
                        j=j+1
                        #EXX=re.search(r'\.*Ex:\s*(.*)',line);
                        DEF=re.search(r'\.*Default:\s*(.*)',line);
                        ENDCOM=re.search(r'\.*\*\/\s*',line);
                        line=re.sub(r"[\t\n]*","",line)
                        line=line.replace("*/","")
                        line=line.replace("/*","")
                        #if (EXX):
                        #    Example[ind] = Example[ind] + '<br>' + EXX[1]
                        if (DEF):
                            Default=Default + '<br>' + DEF[1]
                        else:
                            if (re.search(r'\w',line)):
                                Comment = Comment + '<br>' + line
                        if (ENDCOM):
                            j=-1
        if com:
            Comment=Comment + '<br>' + com[2]
            Default=Default + '<br>' + com[0]
        ParamTable.Reference.append(refword)
        ParamTable.Keys.append(key)
        ParamTable.Default.append(Default)
        ParamTable.Comment.append(Comment)
        ###
    if (refword in myForcings):
        print('Forcing')
        Domain='Forcing'
        com=[]
        for i in range(len(F_lines)):
            found=re.findall(rf"\.*{re.escape(refword)}\s*;", F_lines[i])
            #print(i)
            if (len(found) > 0) and (re.search(r'\s*\/\*', F_lines[i+1])):
                j=1
                Default=''
                Example=''
                Comment=''
                while j > 0:
                    #print(j)
                    line=F_lines[i+j]
                    j=j+1
                    EXX=re.search(r'\.*Ex:\s*(.*)',line);
                    DEF=re.search(r'\.*Default:\s*(.*)',line);
                    ENDCOM=re.search(r'\.*\*\/\s*',line);
                    line=re.sub(r"[\t\n]*","",line)
                    line=line.replace("*/","")
                    line=line.replace("/*","")
                    if (EXX):
                        Example = Example + '<br>' + EXX[1]
                    elif (DEF):
                        Default=Default + '<br>' + DEF[1]
                    else:
                        if (re.search(r'\w',line)):
                            Comment = Comment + '<br>' + line
                    if (ENDCOM):
                        j=-1
        ForcingTable.Reference.append(refword)
        ForcingTable.Keys.append(key)
        ForcingTable.Default.append(Default)
        ForcingTable.Example.append(Example)
        ForcingTable.Comment.append(Comment)
    if not ((refword in myForcings) or (refword in myParams)):
        print('NonId')
        NonIdTable.Reference.append(refword)
        NonIdTable.Keys.append(key)



#%% Creation of the output file

#Creating the mark-down file/table for the list of the user input parameters
Out=open(ParamListFile,'w')
Out.write('# Paramter and Forcing list for BG_Flood\n\n')
Out.write('BG_flood user interface consists in a text file, associating key words to user chosen parameters and forcings.\n')

#Creation of the Parameter table in MD
#####Paramters
Out.write('## List of the Parameters\' input\n\n')
Out.write('|_Reference_|_Keys_|_default_|_Explanation_|\n')
Out.write('|---|---|---|---|\n')

for ind in range(len(ParamTable.Reference)):
    mystr= "|" + str(ParamTable.Reference[ind]) + "|" + str(ParamTable.Keys[ind]) + "|" + str(ParamTable.Default[ind][4:]) + "|" + str(ParamTable.Comment[ind][4:]) + "|\n"
    Out.write(mystr)
Out.write('---\n\n')

#Creation of the Forcing table in MD
#####Forcings
Out.write('## List of the Forcings\' inputs\n\n')
Out.write('|_Reference_|_Keys_|_default_|_Example_|_Explanation_|\n')
Out.write('|---|---|---|---|---|\n')

for ind in range(len(ForcingTable.Reference)):
    mystr= "|" + str(ForcingTable.Reference[ind]) + "|" + str(ForcingTable.Keys[ind]) + "|" + str(ForcingTable.Default[ind][4:]) + "|" + str(ForcingTable.Example[ind][4:]) + "|"+ str(ForcingTable.Comment[ind][4:]) + "|\n"
    Out.write(mystr)
Out.write('---\n\n')


#Creation of the non-identified entries table in MD
#####Non-identified
Out.write('## List of the non-identified inputs\n\n')
Out.write('|_Reference_|_Keys_|\n')
Out.write('|---|---|\n')

for ind in range(len(NonIdTable.Reference)):
        mystr= "|" + str(NonIdTable.Reference[ind]) + "|" + str(NonIdTable.Keys[ind]) + "|\n"
        Out.write(mystr)
Out.write('---\n\n')

Out.write('*Note* : The keys are not case sensitive.\n')


Out.close()


