# remove artifacts

echo "[!] removing artifacts"

if [-d "frames/" ]; then
    echo "[-] rm -r frames"
    rm -r frames
fi

if [-d "partitions/" ]; then
    echo "[-] rm -r partitions/"
    rm -r partitions
fi

if [-d "rsz/" ]; then
    echo "[-] rm -r rsz/"
    rm -r rsz
fi

rm partitions.csv
rm partitions_cons.csv
