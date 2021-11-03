#!/bin/bash
# Script identifying all the key words refering to input parameters in BG_flood and creating
# a list of them in a markdown file
#


echo Creating the list of keys words for the BG_flood interface

## Localisation of the input and output files

BG_ReadInput="src/ReadInput.cu"
BG_Params_h="src/Param.h"
BG_Forcing_h="src/Forcing.h"
ParamListFile="ParametersList.md"


## Get the list of Key parameters

##grep -E -o "^\s*\\{0}\s*parameterstr\s*=\s*\"(\w+)\";" $BG_ReadInput
##grep -oP  "^\s*parameterstr\s*=\s*\"\K\w+\"\s*;\s*$" $BG_ReadInput
#grep -oP  "^\s*parameterstr\s*=\s*\"\K\w+(?=\"+\s*;\s*$)" $BG_ReadInput 
##grep -oP  "^\s*paramvec\s*=\s*\{(\s*\"\s*\w\s*\"\s*\,?)+\"+\s*\}\s*;\s*$" $BG_ReadInput
#grep -oP  "^\s*paramvec\s*=\s*{\K(\s*\"\s*\w+\s*\"\s*\,?)+(?=\s*\}\s*;\s*$)" $BG_ReadInput


####################################################################################
### Read the information from the ReadInput file

###Keys words
keys="$(grep -oP  '(^\s*parameterstr\s*=\s*\K(\"\s*\w+\s*\")(?=\s*;\s*$))|(^\s*paramvec\s*=\s*{\s*\K(\"\s*\w+\s*\"\s*\,?)+(?=\s*\}\s*;\s*$))' $BG_ReadInput)"
#echo "\n \n \n \n Keys \n $keys"

###References: first key word
#refs="$(grep -oP  '(^\s*parameterstr\s*=\s*\K(\"\s*\w+\s*\")(?=\s*;\s*$))|(^\s*paramvec\s*=\s*{\s*\K(\"\s*\w+\s*\")}(?=\s*\,\s*\"\s*\w+\s*\"\s*\,?)+(\s*;\s*$))' $BG_ReadInput)"
#echo "\n\n\n\nrefs\n$refs"

###List of Parameter variables in the code
params="$(grep -oP  '(^\s*param\.\K(\w+)(?=.*))|(^\s*XParam\.\K(\w+)(?=.*))' $BG_ReadInput)"
#echo "\n\n\n\nParams\n$Params"

###List of Forcing parameters in the code
forcings="$(grep -oP  '(^\s*forcing\.\K(\w+)(?=.*))|(^\s*XForcing\.\K(\w+)(?=.*))' $BG_ReadInput)"
#echo "\n\n\n\nForcings\n$Forcings"


#####################################################################################
### Transform the variable in vector

##not working
#function ToVector(){
#
#	local index=0
#	while read -r line
#	do
#		myVar[$index]=$line
#		index=$(($index+1))
#	done <<< "$1"
#	echo "$myVar"
#}
#
#myKeys=$(ToVector $keys)
##end not working

index=0
while read -r line
do
	mykeys[$index]=$line
	ref="$(cut -d',' -f1 <<< "$line")"
	myrefs[$index]=${ref:1:-1}
	index=$(($index+1))
done <<< "$keys"
#echo "len of myKeys: $index"

#index=0
#while read -r line
#do
#        myrefs[$index]=$line
#        index=$(($index+1))
#done <<< "$refs"
#echo "len of myrefs: $index"

index=0
while read -r line
do
	myparams[$index]=$line
	index=$(($index+1))
done <<< "$params"

index=0
while read -r line
do
    myforcings[$index]=$line
    index=$(($index+1))
done <<< "$forcings"


#######################################################################
## Conserve only unique values in Params and Forcings:

declare -A seen
for word in ${myforcings[@]}
do
	if [ ! "${seen[$word]}" ]
	then
		Fs+=("$word")
		seen[$word]=1
	fi
done
#echo "Forcing"
#echo "${myforcings[@]}"
#echo "Forcings simplified"
#echo "${Fs[@]}"


declare -A seen2
for word in ${myparams[@]}
do
	if [ ! "${seen2[$word]}" ]
	then
		Ps+=("$word")
		seen2[$word]=1
	fi
done
#echo "Params simplified"
#echo "${Ps[@]}"


######################################################################################
## Get for each reference the domain (Param/Forcing) and the associated comments and units

#index=0
#domain=()
#comments=()
#for refword in "${myrefs[@]}"
#do
#	echo "domain for $refword :"
#	if [[ "${Ps[*]}" =~ "${refword}" ]]; then
#		echo "In Params"
#	#	domain[$index] ="Param."
#		domain+=( "Param." )
#		com="$(grep $refword $BG_Params_h)"
#		com2="$(cut -d'/' -f3 <<< "${com}")"
#		comments+=( $com2 )
#		#comments+=( "$(grep $refword $BG_Params_h)" )
#		#comments+=( "$(grep -oP  "^\s*$refword\s*=\s*;\s*//\K\w+\s*$" $BG_Params_h)" )
#	elif [[ "${Fs[*]}" =~ "${refword}" ]]; then
#		echo "In Forcing"
#	#	domain[$index] = "Forcing."
#		domain+=( "Forcing." )
#		comments+=( "FFF" )
#	else
#		echo "Not found"
#	#	domain[$index] = "Nan"
#		domain+=( "Nan." )
#		comments+=( "NNNN" )
#	fi
#	index=$(($index+1))
#done

#########################################################################################
## Create the output Markdown file ########

echo "# Paramter and Forcing list for BG_Flood"  >> $ParamListFile
echo " " >> $ParamListFile
echo "BG_flood user interface consists in a text file, associating key words to user chosen parameters and forcings."  >> $ParamListFile
echo " " >> $ParamListFile

##Creation of the variables for the tables
for ind in "${!myrefs[@]}"
do
	refword=${myrefs[$ind]}
	default=( " " )
	comment=( " " )
	anexample=( " " )
	echo "domain for $refword :"
	if [[ "${Ps[*]}" =~ "${refword}" ]]; then
		echo "In Params"
	#	domain[$index] ="Param."
		domain=( "Param." )
		echo "$refword"
	#	com="$(grep -F "^\s*${refword}\.*=\s*$" $BG_Params_h)"	
		com="$(grep "\ ${refword}\s*=" $BG_Params_h)"
		echo "com : $com"
		comment="$(cut -d'/' -f3 <<< "${com}")"
		echo "comment : $comment"
		def="$(cut -d '=' -f2 <<< "${com}")" 
		default="$(cut -d ';' -f1 <<< "${def}")"
	        #toto= -e "$BG_Params_h" | grep "\s${refword}\s*="	
		#echo "toto : $toto"
		#forcings="$(grep -oP  '(^\s*forcing\.\K(\w+)(?=.*))|(^\s*XForcing\.\K(\w+)(?=.*))' $BG_ReadInput)"
		#def="$(cut -d '=' -f2 <<< "${com}")" 
	#	default="$(grep -oP  '(^\s*$refword\s*=\K(.*)(?=//))' $BG_Param_h)"
		#comment="$(grep -oP  '(^\s*\w*\s*${refword}\s*=.*\s*\/\/\K(.*)(?=.*)$)' $BG_Param_h)"
		#comments+=( $com2 )
		#comments+=( "$(grep $refword $BG_Params_h)" )
		#comments+=( "$(grep -oP  "^\s*$refword\s*=\s*;\s*//\K\w+\s*$" $BG_Params_h)" )
	elif [[ "${Fs[*]}" =~ "${refword}" ]]; then
		echo "In Forcing"
		domain=( "Forcing." )
		echo "$refword"
		com="$(awk '/ deform;/,/\*\//' $BG_Forcing_h)"
		#com="$(awk -F"$BG_Forcing_h" 'BEGIN{}"\ ${refword}\s*=" $BG_Params_h)"
		echo "com : $com"
		comment=( "FFF" )
		default=( "None" )
		anexample=( "EEE" )
	else
		echo "Not found"
		domain=( "Nan." )
		comment=( "NNNN" )
		default=( "NNdef" )
	fi
	mydefault+=("${default[0]}")
	mycomment+=("${comment}")
	mydomain+=("${domain}")
	myexample+=("${anexample}")
	#if [[ "${domain}" == "Param." ]]; then
	#	echo "|${myrefs[$ind]}|${mykeys[$ind]}|${default[0]}|${comment}|" >> $ParamListFile
	#fi
done


#####Paramters
echo "## List of the Parameters">> $ParamListFile
echo "|_Reference_|_Keys_|_default_|_Explanation_|" >> $ParamListFile
echo "|----|---|---|---|" >> $ParamListFile
for ind in "${!myrefs[@]}"
do
	if [[ "${mydomain[$ind]}" == "Param." ]]; then
		echo "|${myrefs[$ind]}|${mykeys[$ind]}|${mydefault[$ind]}|${mycomment[$ind]}|" >> $ParamListFile
	fi
done
echo "---" >> $ParamListFile

##### Forcingd
echo "&nbsp;">> $ParamListFile
echo " ">> $ParamListFile
echo "## List of the Forcings">> $ParamListFile
echo "|_Reference_|_Keys_|_default_|_example_|_Explanation_|" >> $ParamListFile
echo "|----|---|---|---|---|" >> $ParamListFile
for ind in "${!myrefs[@]}"
do
	if [[ "${mydomain[$ind]}" == "Forcing." ]]; then
		echo "|${myrefs[$ind]}|${mykeys[$ind]}|${mydefault[$ind]}|${myexample[$ind]}|${mycomment[$ind]}|" >> $ParamListFile
	fi
done
echo "---" >> $ParamListFile

#####Unidentified parameters
echo "&nbsp;">> $ParamListFile
echo " ">> $ParamListFile
echo "## List of the Unidentificated entries">> $ParamListFile
echo "|_Reference_|_Keys_|" >> $ParamListFile
echo "|----|---|" >> $ParamListFile
for ind in "${!myrefs[@]}"
do
	if [[ "${mydomain[$ind]}" == "Nan." ]]; then
		echo "|${myrefs[$ind]}|${mykeys[$ind]}|" >> $ParamListFile
	fi
done
echo "---" >> $ParamListFile


echo " ">> $ParamListFile
#echo "\n" >> $filename
echo "*Note* : The keys are not case sensitive." >> $ParamListFile


