#!/bin/bash

fix_ranks() {
    #shopt -s extglob
    local target_dir="$1"
    cd $target_dir

    local pyfrmf=$(ls -1a . | grep -i pyfrm$ | head -n1)
    local pyfrsf=$(find solution/ | grep -i pyfrs$ | xargs)
    local pyfrsf=($pyfrsf)
    pyfr partition "$2" "$pyfrmf" "${pyfrsf[@]}" .

    local new_pyfrsf=$(ls -1a . | grep -i pyfrs$ | xargs)
    local new_pyfrsf=($new_pyfrsf)
    mv "${new_pyfrsf[@]}" solution/
    
}

for var in $(find "$1" -maxdepth 1 -mindepth 1 -type d); do
    fix_ranks $var "$2"
done
