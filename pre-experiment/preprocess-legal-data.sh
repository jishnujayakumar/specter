directory=$ELECTER_DIR/$1

echo "Replacement..."
sed -i 's/\ \ =\ \ / : /g' $directory/mapping.txt 
sed -i 's/-->/:/g' $directory/no_doc_mapping.txt 
sed -i 's/,/ : /g' $directory/precedent-citation.txt 
sed -i 's/no_doc_1/1975_T_30/g' $directory/precedent-citation.txt 
sed -i 's/no_doc_4/2005_P_74/g' $directory/precedent-citation.txt 
sed -i 's/no_doc_3/2000_P_55/g' $directory/precedent-citation.txt 
sed -i 's/no_doc_6/2011_C_81/g' $directory/precedent-citation.txt 
sed -i 's/no_doc_5/2006_A_94/g' $directory/precedent-citation.txt 
sed -i 's/no_doc_2/1961_C_3/g' $directory/precedent-citation.txt 
sed -i 's/no_doc_8/2001_S_302/g' $directory/precedent-citation.txt 
sed -i 's/no_doc_7/2013_K_16/g' $directory/precedent-citation.txt
echo "Replacement Done."

echo "Preprocessing..."
python3 ./pre-experiment/preprocess.py $directory
echo "Preprocessing Done."
