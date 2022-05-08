To render documentation.md to pdf use the following:
```
sudo apt update
sudo apt install nodejs npm

npx embedme --strip-embed-comment --stdout docs/documentation.md > docs/tmp.md && pandoc --standalone --toc  docs/tmp.md --pdf-engine=xelatex --resource-path=docs -s --highlight-style kate -o docs/pdf-documentation.pdf

npx embedme --strip-embed-comment --stdout docs/documentation.md > docs/tmp.md && pandoc --standalone --toc docs/tmp.md --pdf-engine=xelatex --resource-path=docs -s --wrap auto --highlight-style espresso -o docs/pdf-documentation.pdf -H docs/head.tex

pandoc --standalone --toc README.md --pdf-engine=xelatex --resource-path=docs -s --wrap auto --highlight-style espresso -o pdf-technical-documentation.pdf -H docs/head.tex
```

To resize images to 2000px:
```
mogrify -resize 2000x *.jpg *.png *.jpeg
```