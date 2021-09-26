arr=(`locate settings.json`)
#src1="/usr/local/cuda-10.1/targets/x86_64-linux/include/**"
src2="/usr/local/cuda-10.1/targets/x86_64-linux/include"
# src="/usr/local/cuda-10.1/include/**/**"
dst="/usr/local/cuda-10.1/include/"
for file in ${arr[@]}; do
    # sed -i 's?'${src}'?'${dst}'?g' ${file}
    #sed -i 's?'${src1}'?'${dst}'?g' ${file}
    sed -i 's?'${src2}'?'${dst}'?g' ${file}
done

echo "Substitution completed."
