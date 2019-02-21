while getopts i:o: option
do
case "${option}"
in
i) INPUT=${OPTARG};;
o) OUTPUT=${OPTARG};;
esac
done
export PATH=anaconda3/bin:$PATH
python final_main_1.py -i $INPUT -o $OUTPUT
python final_main_2.py -i $INPUT -o $OUTPUT
python final_main_3.py -i $INPUT -o $OUTPUT
python final_main_4.py -i $INPUT -o $OUTPUT
python final_main_5.py -i $INPUT -o $OUTPUT
python final_main_6.py -i $INPUT -o $OUTPUT
python final_main_7.py -i $INPUT -o $OUTPUT
python final_main_8.py -i $INPUT -o $OUTPUT