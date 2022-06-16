#!/bin/bash

command_exists() {
	if ! [ -z "$(command -v $1)" ]; then
		echo 1
	else
		echo 0
	fi 
}

run_case() {
	local mesh_file="$1"
	local mesh_file_basename=$(basename "${mesh_file%.*}")
	local mesh_file_ext=${mesh_file##*.}
	local config_file=$(realpath "$2")
	local output_base_dir=${3:-./}

	#choose backend
	if [ $(command_exists nvidia-smi) -eq 1 ]; then
		local n_gpus=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)
		local pyfr_backend="cuda"
	elif [ $(command_exists rocm-smi) -eq 1 ]; then
		local n_gpus=$(rocm-smi --showuniqueid | grep Unique | grep -v "=" | wc -l)
		local pyfr_backend="hip"
	elif [ $(command_exists clinfo) -eq 1 ]; then
		local n_gpus=$(clinfo -l | grep -i device | wc -l)
		local pyfr_backend="opencl"
	else
		local n_gpus="1"
		local pyfr_backend="openmp"
	fi
	
	local output_dir="${output_base_dir}/${mesh_file_basename}"
	mkdir -p "${output_dir}/solution"
	cp ${mesh_file} ${output_dir}
	
	local pyfrm_path="${output_dir}/${mesh_file_basename}.pyfrm"
	local pyfrm_base_path=$(basename "${pyfrm_path}")
	if [ $mesh_file_ext = "msh" ];then
		pyfr import -tgmsh ${mesh_file} ${pyfrm_path}
		pyfr partition ${n_gpus} ${pyfrm_path} ${output_dir}
	fi

	pushd ${output_dir}
	local pyfrm_base_path=$(basename "${pyfrm_path}")
	echo "running ${pyfrm_base_path}"
	OMPI_ALLOW_RUN_AS_ROOT=1 OMPI_ALLOW_RUN_AS_ROOT_CONFIRM=1 mpirun -np ${n_gpus} pyfr run --backend ${pyfr_backend} --progress ${pyfrm_base_path} ${config_file} > pyfr.log
	popd
}

cfg_file="/dev/null"
output_base_dir="./"
processed_args=0

for var in "$@"; do
	if [ $processed_args -eq 0 ]; then
		cfg_file="$var"
	elif [ $processed_args -eq 1 ] && [[ -d $var ]]; then
		output_base_dir="$var"
	else
		run_case $var $cfg_file $output_base_dir
	fi
	
	processed_args=$(( processed_args + 1 ))
done
