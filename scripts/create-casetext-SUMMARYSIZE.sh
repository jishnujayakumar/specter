#!/bin/bash
# Usage: ./create-casetext-SUMMARYSIZE.sh legal-data-dsdr-summarized 512 (name of legal-data dir, summarySize)
dir="$ELECTER_DIR/$1"
outputF="$dir/file-to-summary-size.txt"
inpDir="$dir/casetext"
summaryTokenSize=$2

rm outputF

for casetextFile in `ls $inpDir`;do
    echo $casetextFile$'\t'$summaryTokenSize >> $outputF
done

mv $inpDir/ $dir/original-castext-without-summarization/