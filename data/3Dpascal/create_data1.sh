cur_dir=$(cd $( dirname ${BASH_SOURCE[0]} ) && pwd )
root_dir=$cur_dir/../../
cd $root_dir

redo=false
dataset_name="pascal3D"
data_root_dir="$cur_dir/$dataset_name"
mapfile="$cur_dir/$dataset_name/labelmap_3D.prototxt"
anno_type="detection"
label_type="json"
db="lmdb"
min_dim=0
max_dim=0
width=500
height=500

extra_cmd="--encode-type=jpg --encoded --redo"

for subset in train 
do
  python $root_dir/scripts/create_annoset.py --anno-type=$anno_type --label-type=$label_type --label-map-file=$mapfile --min-dim=$min_dim --max-dim=$max_dim $extra_cmd --root=$data_root_dir --listfile=$data_root_dir/cache/$subset/$subset.txt --outdir=$data_root_dir/$db/$dataset_name"_"$subset"_"$db examples/$dataset_name 2>&1 | tee $cur_dir/$dataset_name/$subset.log
done
