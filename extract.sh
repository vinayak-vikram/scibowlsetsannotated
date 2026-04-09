#!/bin/zsh
for file in $SETSDIR/*.pdf; do
    echo "Extracting $file..."
    $PDFEXTRACT -f $file >> $SETSDIR/$(basename $file .pdf).txt
done
